import numpy
score = numpy.array(0.672970)
account = {
'BTC_10_25_1_0.1': {
    'Cryptocurrency': 'ETH',
    'Score': score,
    'Params':[{'window_size': 25,
                'units_2': 32,
                'units_1': 64,
                'optimizer': 'rmsprop',
                'n_state_units': 128,
                'activation_2': 'sigmoid',
                'activation_1': 'relu'}],
           }}

f = open('../evaluate_result/BTC_30_25_1_0.1_param.pickle', 'wb')
import pickle
pickle.dump(account, f)
f.close()