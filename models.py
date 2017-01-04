from keras.layers import Dense, Input, Dropout, merge
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.regularizers import l2, activity_l2
from gym import spaces

def rnn(input_shape, time_steps, num_h, layers):
    # Build Network
    # Dropout hyper-parameters based on Hinton's paper
    inputs = x = Input(shape=(time_steps,) + input_shape, name='input')
    x = Dropout(0.2)(x)
    x = LSTM(num_h, activation='relu', name='hidden')(x)
    x = Dropout(0.5)(x)

    for i in range(layers - 1):
        x = Dense(num_h, activation='relu', name='hidden' + str(i))(x)
        x = Dropout(0.5)(x)

    return Model(inputs, x)

def dense(input_shape, num_h, layers, dropout=0, regularization=0):
    # Build Network
    inputs = x = Input(shape=input_shape, name='input')
    if dropout != 0:
        x = Dropout(dropout)(x)

    for i in range(layers):
        x = Dense(num_h,
                  activation='relu',
                  name='hidden' + str(i),
                  W_regularizer=l2(regularization),
                  activity_regularizer=activity_l2(regularization)
        )(x)

        if dropout != 0:
            x = Dropout(dropout)(x)

    return inputs, x

def rnn_1(input_space, time_steps):
    # Build Network

    # Build multiple inputs. One for each tuple
    inputs = []
    for i, space in enumerate(input_space):
        # One hot vector if it's discrete. Otherwise take the shape of Box.
        shape = (space.n,) if isinstance(space, spaces.Discrete) else space.shape
        inputs.append(Input(shape=(time_steps,) + shape, name='input' + str(i)))

    x = merge(inputs, mode='concat', concat_axis=2)
    x = LSTM(128, activation='relu', name='lstm')(x)
    x = Dense(128, activation='relu', name='hidden1')(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dense(512, activation='relu', name='hidden3')(x)
    #x = Dense(512, activation='relu', name='hidden4')(x)
    return inputs, x
