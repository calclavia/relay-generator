import numpy as np
from keras.layers import Dense, Input, merge, Activation, Flatten
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution2D
from keras.models import Model
from keras.regularizers import l2, activity_l2


def relay_dense(input_space):
    # Build Network
    map_space, pos_shape, dir_shape, difficulty_shape = input_space.spaces
    num_block_types = int((map_space.high - map_space.low).max())

    # Define inputs
    block_input = Input(
        shape=(
            map_space.shape[0],
            map_space.shape[1],
            num_block_types
        ),
        name='block_input'
    )
    pos_input = Input(shape=pos_shape.shape, name='pos_input')
    dir_input = Input(shape=(dir_shape.n,), name='dir_input')
    difficulty_input = Input(
        shape=difficulty_shape.shape, name='difficulty_input')

    # Build image processing
    image = block_input
    image = Convolution2D(32, 3, 3, name='conv1')(image)
    image = Activation('relu')(image)
    image = Convolution2D(64, 3, 3, name='conv2')(image)
    image = Activation('relu')(image)
    image = Convolution2D(128, 3, 3, name='conv3')(image)
    image = Activation('relu')(image)
    image = Convolution2D(256, 3, 3, name='conv4')(image)
    image = Activation('relu')(image)

    image = Flatten()(image)

    # Build feature processing
    feature = merge([pos_input, dir_input,  difficulty_input],
                    mode='concat', concat_axis=1)

    # Merge all features
    x = merge([image, feature], mode='concat')

    for i in range(3):
        x = Dense(256, name='h1_' + str(i))(x)
        x = Activation('relu')(x)

    for i in range(2):
        x = Dense(512, name='h2_' + str(i))(x)
        x = Activation('relu')(x)

    return [block_input, pos_input, dir_input, difficulty_input], x


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
