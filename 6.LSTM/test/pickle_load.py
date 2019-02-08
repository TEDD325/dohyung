import pickle
f = open('../evaluate_result/BTC_10_25_1_0.1_param.pickle', 'rb')
a = pickle.load(f)
print(a)
print(type(a))
print(a.keys())