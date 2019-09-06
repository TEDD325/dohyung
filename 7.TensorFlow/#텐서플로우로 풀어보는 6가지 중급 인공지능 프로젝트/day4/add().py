from keras.layers import Input, Dense, Add
from keras.models import Model

input1 = Input(shape = (16,))
x1 = Dense(8)(input1)
input2 = Input(shape = (32,))
x2 = Dense(8)(input2)
added = Add()([x1, x2])
print('Add shape', added.shape)

out = Dense(4)(added)
model = Model(inputs = [input1, input2], outputs = out)
model.summary()