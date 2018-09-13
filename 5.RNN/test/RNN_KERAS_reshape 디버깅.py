
# coding: utf-8

# In[ ]:


# key_name_X = "X_" + \
#                 str(time_unit[0]) + "_" + \
#                 str(window_size[4]) + "_" + \
#                 str(gap[0]) + "_" + \
#                 str(margin_rate[0])

# key_name_y = "y_" + \
#                 str(time_unit[0]) + "_" + \
#                 str(window_size[4]) + "_" + \
#                 str(gap[0]) + "_" + \
#                 str(margin_rate[0])


# with open(dataset_dir_path_tuple_type + key_name_X + ".pickle", 'rb') as handle:
#     X = pickle.load(handle)
# with open(dataset_dir_path_tuple_type + key_name_y + ".pickle", 'rb') as handle:
#     y = pickle.load(handle)

# evaluate_result_dir_path = "./evaluate_result/"
# dataset_dir_path = dataset_dir_path_tuple_type

# Evaluate(dataset_dir_path, evaluate_result_dir_path, time_unit, window_size, gap, margin_rate)

# key_name_X = "X_" + \
#                 str(time_unit[0]) + "_" + \
#                 str(window_size[0]) + "_" + \
#                 str(gap[0]) + "_" + \
#                 str(margin_rate[0])

# key_name_y = "y_" + \
#                 str(time_unit[0]) + "_" + \
#                 str(window_size[0]) + "_" + \
#                 str(gap[0]) + "_" + \
#                 str(margin_rate[0])


# with open(dataset_dir_path_tuple_type + key_name_X + ".pickle", 'rb') as handle:
#     X = pickle.load(handle)
# with open(dataset_dir_path_tuple_type + key_name_y + ".pickle", 'rb') as handle:
#     y = pickle.load(handle)

# X.shape

# y_single = {}
# y_single['BTC'] = y[:, 1]
# y_single['ETH'] = y[:, 2]
# y_single['XRP'] = y[:, 3]
# y_single['BCH'] = y[:, 4]
# y_single['LTC'] = y[:, 5]
# y_single['DASH'] = y[:, 6]
# y_single['ETC'] = y[:, 7]

# coin_list2 = ["BTC", "ETH", "XRP", "BCH", "LTC", "DASH", "ETC"]

# for coin in coin_list2:
#     print(y_single[coin].shape)

# X_train_2 = X_train.transpose([0, 2, 1, 3])
# X_test_2 = X_test.transpose([0, 2, 1, 3])
# print("X_train_2.shape")
# print(X_train_2.shape)
# print("X_test_2.shape")
# print(X_test_2.shape)
# print()

# tmp = 1
# for i in range(4):
#     tmp *= X_train_2.shape[i]
# print(tmp)

# n_steps

# X_train_3 = X_train_2.reshape([X_train.shape[0], n_steps, n_coins * n_price])
# X_test_3 = X_test_2.reshape([X_test.shape[0], n_steps, n_coins * n_price])
# print("X_train_3.shape")
# print(X_train_3.shape)
# print("X_test_3.shape")
# print(X_test_3.shape)
# print()


# In[ ]:


# y_single = {}
# y_single['BTC'] = y[:, 1]
# y_single['ETH'] = y[:, 2]
# y_single['XRP'] = y[:, 3]
# y_single['BCH'] = y[:, 4]
# y_single['LTC'] = y[:, 5]
# y_single['DASH'] = y[:, 6]
# y_single['ETC'] = y[:, 7]

# coin_list2 = ["BTC", "ETH", "XRP", "BCH", "LTC", "DASH", "ETC"]

# for coin in coin_list2:
#     print("y_single["+coin+"]"+".shape")
#     print(y_single[coin].shape)
#     print()

# # y2 = onehottify(y_single['BTC'], n=2)
# y2 = {}
# for coin in coin_list2:
#     y2[coin] = onehottify(y_single[coin], n=2)


#     X_train, X_test, y_train, y_test = train_test_split(X, y2[coin], test_size=0.1, random_state=42)
#     # y2[coin]으로 수정
#     print("X_train.shape")
#     print(X_train.shape)
#     print("y_train.shape")
#     print(y_train.shape)
#     print()
#     print("X_test.shape")
#     print(X_test.shape)
#     print("y_test.shape")
#     print(y_test.shape)
#     print()

#     n_coins = 8
#     n_price = 4
#     n_steps = window_size[0]

#     X_train_2 = X_train.transpose([0, 2, 1, 3])
#     X_test_2 = X_test.transpose([0, 2, 1, 3])
#     print("X_train_2.shape")
#     print(X_train_2.shape)
#     print("X_test_2.shape")
#     print(X_test_2.shape)
#     print()

#     X_train_3 = X_train_2.reshape([X_train.shape[0], n_steps, n_coins * n_price])
#     X_test_3 = X_test_2.reshape([X_test.shape[0], n_steps, n_coins * n_price])
#     print("X_train_3.shape")
#     print(X_train_3.shape)
#     print("X_test_3.shape")
#     print(X_test_3.shape)
#     print()


#     param_grid = {
#         'window_size' : [n_steps], 
#         'n_state_units': [40, 80, 160],
#         'activation': ['relu', 'softmax'], 
#         'optimizer': ['sgd', 'rmsprop', 'adam'], #sgd 추가
#         'init': ['glorot_uniform', 'normal', 'uniform', 'he'], #he 추가
#         'batch_size': [10, 50] 
#     }

#     X_train_reshape = X_train_2.reshape([X_train.shape[0], n_steps*n_coins * n_price])
#     X_test_reshape = X_test_2.reshape([X_test.shape[0], n_steps*n_coins * n_price])
#     print("X_train_reshape.shape")
#     print(X_train_reshape.shape)
#     print("X_test_reshape.shape")
#     print(X_test_reshape.shape)
#     print()

#     scaler = MinMaxScaler()
#     scaler.fit(X_train_reshape)
#     X_train_scaled = scaler.transform(X_train_reshape)
#     X_test_scaled = scaler.transform(X_test_reshape)

#     X_train_scaled = X_train_scaled.reshape(-1, n_steps, n_coins * n_price)

#     X_test_scaled = X_test_scaled.reshape(-1, n_steps, n_coins * n_price)

#     model = KerasClassifier(
#         build_fn=create_model, 
#         epochs=1, # epochs는 실험을 최종적으로 수행하고자 할 때 높일 것(100~150정도)
#         batch_size=10, 
#         verbose=True
#     )

#     grid = GridSearchCV(
#         estimator=model, 
#         cv=5, 
#         param_grid=param_grid
#     )

#     X_train_scaled, X_test_scaled = input_reshape(X_train_scaled, X_test_scaled)

#     print("----------------------")
#     print("time unit: "+str(time_unit[0]) + "  |  " + \
#           "window_size :"+str(window_size[4]) + "  |  " + \
#           "gap :"+str(gap[0]) + "  |  " + \
#           "margin_rate :"+str(margin_rate[0]) + \
#           "  started.")
#     grid_result = grid.fit(X_train_scaled, y_train)
#     print("----------------------")

#     evaluate_result = {}
#     evaluate_result['grid_result.best_score_'] = grid_result.best_score_
#     evaluate_result['grid_result.best_params_'] = grid_result.best_params_

#     # summarize results
#     print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#     print("pickle saving..")
#     with open(pickle_result_dir_path + key_name_X+"_result.pickle", 'wb') as handle:
#         pickle.dump(evaluate_result[key_name_X], handle, protocol=pickle.HIGHEST_PROTOCOL)
#     print()

#     means = grid_result.cv_results_['mean_test_score']
#     stds = grid_result.cv_results_['std_test_score']
#     params = grid_result.cv_results_['params']

#     for mean, stdev, param in zip(means, stds, params):
#         print("%f (%f) with: %r" % (mean, stdev, param))
#     print()
    
#     pyplot.plot(history.history['rmse']) # rmse 자리에는 metrics에 쓰인 함수의 이름을 적는다. epoch마다 loss값의 감소 상황을 관찰할 수 있다.
#     pyplot.show()
#     plt.savefig(pickle_result_dir_path+key_name_X+'_result.png')

