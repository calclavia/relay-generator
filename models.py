import numpy as np
from keras.layers import Dense, Input, merge, Activation, Flatten
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution2D
from keras.models import Model
#from keras.layers.normalization import BatchNormalization

def conv(filters, res=None, border_mode='same'):
    def f(x):
        x = Convolution2D(filters, 3, 3, border_mode=border_mode)(x)
        if res is not None:
            x = merge([x, res], mode='sum')

        #x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    return f

def dense(units):
    def f(x):
        x = Dense(units)(x)
        #x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    return f

def relay_dense(input_space, num_actions):
    # Build Network
    map_space, pos_shape, dir_shape, difficulty_shape = input_space.spaces
    num_block_types = int((map_space.high - map_space.low).max())

    # Define inputs
    block_input = Input(
        (
            map_space.shape[0],
            map_space.shape[1],
            num_block_types
        ),
        name='block_input'
    )
    pos_input = Input(pos_shape.shape, name='pos_input')
    dir_input = Input((dir_shape.n,), name='dir_input')
    difficulty_input = Input(difficulty_shape.shape, name='difficulty_input')

    # Build image processing
    image = block_input

    image = conv(16, border_mode='valid')(image)
    image = conv(16, border_mode='valid')(image)
    image = conv(32, border_mode='valid')(image)
    image = conv(32, border_mode='valid')(image)

    image = Flatten()(image)

    # Build context feature processing
    context = merge([pos_input, dir_input, difficulty_input],
                    mode='concat', concat_axis=1, name='context')
    context = dense(512)(context)

    x = dense(512)(image)
    x = merge([x, context], mode='sum')
    x = dense(512)(x)
    x = merge([x, context], mode='sum')
    x = dense(512)(x)

    # Multi-label
    policy = Dense(num_actions, name='policy', activation='softmax')(x)
    value = Dense(1, name='value', activation='linear')(x)

    model = Model([block_input, pos_input, dir_input,
                   difficulty_input], [policy, value])
    return model


def relay_preprocess(env, observation):
    """
    Preprocesses the relay space
    """
    # TODO: Could be optimized?
    map_space, pos_shape, dir_shape, difficulty_shape = env.observation_space.spaces
    num_block_types = int((map_space.high - map_space.low).max())

    map_state, pos_state, dir_state, difficulty_state = observation
    # Turn map_state into a one-hot channel bit image
    map_state = (np.arange(num_block_types) == map_state[:, :, None] - 1)
    # One hot the dir_state
    dir_state_hot = np.zeros((dir_shape.n,))
    dir_state_hot[dir_state[0] + 1] = 1
    return (map_state, pos_state, dir_state_hot, difficulty_state)
