import numpy as np
state = np.array([150, -1.2, -120, 3.2])
print(state)
print(state.shape)
state_size = 4
state = np.reshape(state, [1, state_size, 1, 1])
print(state)
print(state.shape)