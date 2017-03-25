import numpy as np
from keras.layers import Dense, Input, Concatenate, Add, Activation, Flatten, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv2D
from keras.models import Model
from relay_generator import BlockType

def relay_dense(input_space, num_actions, units=150):
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
    image = Dropout(0.2)(image)

    for l in range(3):
        prev = image
        image = Conv2D(32, 3, padding='same')(image)
        image = Activation('relu')(image)
        image = Dropout(0.5)(image)

        # Residual connection
        if l > 0:
            image = Add()([image, prev])

    image = Flatten()(image)

    # Build context feature processing
    context = Concatenate(name='context')([pos_input, dir_input, difficulty_input])
    context = Dense(32)(context)
    context = Activation('relu')(context)
    context = Dropout(0.5)(context)

    out = Concatenate()([image, context])

    policy = Dense(num_actions, name='policy', activation='softmax')(out)
    value = Dense(1, name='value', activation='linear')(out)

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

    # Turn map_state into a one-hot bit image
    map_state = np.reshape(np.sign(map_state), map_state.shape + (1,))

    # One hot the dir_state
    dir_state_hot = np.zeros((dir_shape.n,))
    dir_state_hot[dir_state[0] + 1] = 1
    return (map_state, pos_state, dir_state_hot, difficulty_state)
