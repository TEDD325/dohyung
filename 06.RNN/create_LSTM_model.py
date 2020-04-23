from local_function import *
from start_model import *

def create_model_LSTM(window_size, n_state_units=32, activation='softmax', optimizer='adam', init='glorot_uniform', dropout_rate=0.0, neurons=2):
    model = Sequential()
    model.add(# if문을 통해 여러 RNN모델 쓸 수 있도록 하기, SimpleRNN외에 다른 RNN모델 찾아보기
        LSTM( n_state_units, 
              input_shape=(window_size, 32),
              use_bias=True, 
              activation='tanh',
              kernel_initializer='glorot_uniform', 
              recurrent_initializer='orthogonal', 
              bias_initializer='zeros', 
              dropout=0.0,
              recurrent_dropout=0.0))
    
    model.add(Dense(units=neurons))
    model.add(Dropout(dropout_rate))
#     model.add(Dense(units=2))
    model = multi_gpu_model(model, gpus=2)

    model.compile(loss=keras.losses.categorical_crossentropy, 
                  optimizer=optimizer, 
                  metrics=["accuracy", f1_score])

    return model


def create_model_LSTM_non_GPU(window_size, 
                              n_state_units=32, 
                              activation='softmax', 
                              optimizer='adam'):
    K.clear_session()
    model = Sequential()
    model.add(# if문을 통해 여러 RNN모델 쓸 수 있도록 하기, SimpleRNN외에 다른 RNN모델 찾아보기
        LSTM(units=n_state_units, 
             input_shape=(window_size, 32),
             use_bias=True))
    
    model.add(Dense(units=2))
#     model.add(Dropout(dropout_rate))

    model.compile(loss=keras.losses.categorical_crossentropy, 
                  optimizer=optimizer, 
                  metrics=["accuracy", f1_score])

    return model