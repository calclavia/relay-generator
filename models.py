from keras.layers import Dense, Input, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.regularizers import l2, activity_l2

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

def dense_1(input_shape):
    # Build Network
    # Dropout hyper-parameters based on Hinton's paper
    inputs = x = Input(shape=input_shape, name='input')
#    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu', name='hidden1')(x)
#    x = Dropout(0.1)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
#    x = Dropout(0.25)(x)
    x = Dense(512, activation='relu', name='hidden3')(x)
#    x = Dropout(0.25)(x)
    return inputs, x
