import numpy as np
from keras.layers import Dense, Input, Activation, Flatten, Dropout, merge
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution2D
from keras.models import Model
from keras.regularizers import l2
from relay_generator import BlockType

def relay_dense(input_space, num_actions, reg=1e-3, dropout=0.2):
    # Build Network
    map_space, pos_shape, dir_shape, difficulty_shape = input_space.spaces
    num_block_types = int((map_space.high - map_space.low).max())

    # Define inputs
    block_input = Input(map_space.shape + (1,), name='block_input')
    pos_input = Input(pos_shape.shape, name='pos_input')
    dir_input = Input((dir_shape.n,), name='dir_input')
    difficulty_input = Input(difficulty_shape.shape, name='difficulty_input')

    # Build image processing
    image = block_input
    # image = Dropout(0.1)(image)

    for l, units in enumerate([5, 10, 20]):
        prev = image
        # image = Conv2D(32, 3, padding='same')(image)
        image = Convolution2D(units, 3, 3, W_regularizer=l2(reg), b_regularizer=l2(reg))(image)
        image = Activation('relu')(image)
        # image = Dropout(dropout)(image)

    image = Flatten()(image)

    # Build context feature processing
    context = merge([pos_input, dir_input, difficulty_input], mode='concat')
    # context = Concatenate(name='context')([pos_input, dir_input, difficulty_input])
    context = Dense(10, W_regularizer=l2(reg), b_regularizer=l2(reg))(context)
    context = Activation('relu')(context)
    # context = Dropout(dropout)(context)

    out = merge([image, context], mode='concat')
    # out = Concatenate()([image, context])

    policy = Dense(num_actions, name='policy', activation='softmax', W_regularizer=l2(reg), b_regularizer=l2(reg))(out)
    value = Dense(1, name='value', activation='linear', W_regularizer=l2(reg), b_regularizer=l2(reg))(out)

    model = Model([block_input, pos_input, dir_input,
                   difficulty_input], [policy, value])
    return model


def relay_preprocess(env, observation):
    """
    Preprocesses the relay space
    """
    map_space, pos_shape, dir_shape, difficulty_shape = env.observation_space.spaces
    num_block_types = int((map_space.high - map_space.low).max())

    map_state, pos_state, dir_state, difficulty_state = observation

    # Turn map_state into a one-hot bit image
    map_state = np.reshape(np.sign(map_state), map_state.shape + (1,))

    # One hot the dir_state
    dir_state_hot = np.zeros((dir_shape.n,))
    dir_state_hot[dir_state[0]] = 1
    return (map_state, pos_state, dir_state_hot, difficulty_state)
