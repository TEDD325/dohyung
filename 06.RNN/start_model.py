from local_function import *
def Start_Model(pickle_load_dir_path, 
                pickle_result_dir_path, 
                idx_time_unit, 
                idx_window_size, 
                idx_gap, 
                idx_margin_rate, 
                epochs, 
                MODEL, 
                _GPU, 
                n_jobs,
                cv,
                dataset_scale,
                machine):
    X = {}
    y = {}

    key_name_X = "X_"
    key_name_y = "y_"

    key_name_X += str(idx_time_unit) + "_" + str(idx_window_size) + "_" + str(idx_gap) + "_" + str(idx_margin_rate)
    key_name_y += str(idx_time_unit) + "_" + str(idx_window_size) + "_" + str(idx_gap) + "_" + str(idx_margin_rate)

    # remove [:10000], when real training
    X = Load_Dataset_X(pickle_load_dir_path, idx_time_unit, idx_window_size, idx_gap, idx_margin_rate)[0][:dataset_scale]
    y = Load_Dataset_y(pickle_load_dir_path, idx_time_unit, idx_window_size, idx_gap, idx_margin_rate)[1][:dataset_scale]
    
    y_single = {}
#     print("[INFO] y : {}".format(y))
#     y = np.asarray(y[0])
#     print("[INFO] y.shape : {}".format(y.shape))
#     print("[INFO] y : {}".format(y))
    y_single['BTC'] = y[:, 1]
    y_single['ETH'] = y[:, 2]
    y_single['XRP'] = y[:, 3]
    y_single['BCH'] = y[:, 4]
    y_single['LTC'] = y[:, 5]
    y_single['DASH'] = y[:, 6]
    y_single['ETC'] = y[:, 7]

    coin_list2 = ["BTC", "ETH", "XRP", "BCH", "LTC", "DASH", "ETC"]


    for coin in coin_list2:
        if (os.path.isfile(pickle_result_dir_path + \
                          MODEL + "_" + \
                          coin + "_" + \
                          str(idx_time_unit) + "_" + \
                          str(idx_window_size) + "_" + \
                          str(idx_gap) + "_" + \
                          str(idx_margin_rate) + \
                          "_result.pickle")) is True:
            print(MODEL + "_" + \
                  coin + "_" + \
                  str(idx_time_unit) + "_" + \
                  str(idx_window_size) + "_" + \
                  str(idx_gap) + "_" + \
                  str(idx_margin_rate) + \
                  "_result.pickle FILE ALREADY EXIST.")
            continue
        else:
            y2 = onehottify(y_single[coin], n=2)

            X_train, X_test, y_train, y_test = train_test_split(X, 
                                                                y2, 
                                                                test_size=0.1, 
                                                                random_state=42)
            print("[INFO] X_train.shape : {}".format(X_train.shape))
            print("[INFO] y_train.shape : {}".format(y_train.shape))
            print("[INFO] X_test.shape : {}".format(X_test.shape))
            print("[INFO] y_test.shape : {}".format(y_test.shape))
            print()

            n_coins = 8
            n_price = 4
            n_steps = idx_window_size 

            X_train_2 = X_train.transpose([0, 2, 1, 3])
            X_test_2 = X_test.transpose([0, 2, 1, 3])
            print("[INFO] X_train_2.shape: {}".format(X_train_2.shape))
            print("[INFO] X_test_2.shape: {}".format(X_test_2.shape))
            print()
            
            X_train_3 = X_train_2.reshape([X_train.shape[0], n_steps, n_coins * n_price])
            X_test_3 = X_test_2.reshape([X_test.shape[0], n_steps, n_coins * n_price])
            print("[INFO] X_train_3.shape: {}".format(X_train_3.shape))
            print("[INFO] X_test_3.shape: {}".format(X_test_3.shape))
            print()

            X_train_reshape = X_train_2.reshape([X_train.shape[0], n_steps*n_coins * n_price])
            X_test_reshape = X_test_2.reshape([X_test.shape[0], n_steps*n_coins * n_price])
            print("[INFO] X_train_reshape.shape: {}".format(X_train_reshape.shape))
            print("[INFO] X_test_reshape.shape: {}".format(X_test_reshape.shape))
            print()
            
            param_grid = {'window_size' : [idx_window_size], 
                          'n_state_units': [16, 32, 64],
                          'activation': ['tanh', 'sigmoid', 'relu'], 
                          'optimizer': ['rmsprop', 'Adam', 'Adagrad']}

            scaler = MinMaxScaler()
            scaler.fit(X_train_reshape)
            
            X_train_scaled = scaler.transform(X_train_reshape)
            X_test_scaled = scaler.transform(X_test_reshape)

            X_train_scaled = X_train_scaled.reshape(-1, 
                                                    n_steps, 
                                                    n_coins * n_price)
            X_test_scaled = X_test_scaled.reshape(-1, 
                                                  n_steps, 
                                                  n_coins * n_price)

            if _GPU == True:
                model = KerasClassifier(build_fn=create_model_LSTM, 
                                        epochs=epochs, 
    #                                     batch_size=100, 
                                        verbose=True)
                
            elif _GPU == False:
                model = KerasClassifier(build_fn=create_model_LSTM_non_GPU, 
                                        epochs=epochs, 
                                        batch_size=10, 
                                        verbose=True)

            grid = GridSearchCV(estimator=model, 
                                cv=cv, 
                                n_jobs=n_jobs, # test
                                param_grid=param_grid,
                                verbose=1)

            
            X_train_scaled, X_test_scaled = input_reshape(X_train_scaled,  
                                                          X_test_scaled, 
                                                          n_steps, 
                                                          n_coins, 
                                                          n_price)
                      
            
            print()
            print()
            print("----------------------")
            print("<"+MODEL+">")
            print("----------------------")
            print("__"+coin+"__" + \
                    "time unit: "+str(idx_time_unit) + "  |  " + \
                    "window_size :"+str(idx_window_size) + "  |  " + \
                    "gap :"+str(idx_gap) + "  |  " + \
                    "margin_rate :"+str(idx_margin_rate) + \
                    "  started.")

            grid_result = grid.fit(X_train_scaled, 
                                   y_train, 
                                   validation_data=(X_test_scaled,
                                                    y_test))

            print("----------------------")
            print("grid_result.score(X_test_scaled, y_test): ",grid_result.score(X_test_scaled, y_test))

            evaluate_result = {}

            test_score = grid_result.score(X_test_scaled, y_test)
            evaluate_result[MODEL + "_" + \
                          coin + "_" + \
                          str(idx_time_unit) + "_" + \
                          str(idx_window_size) + "_" + \
                          str(idx_gap) + "_" + \
                          str(idx_margin_rate)] = {"MODEL":MODEL,\
                                            "Cryptocurrency":coin, \
                                            "Score":grid_result.cv_results_['mean_test_score'], \
                                            "Params":grid_result.cv_results_['params'],\
                                            "test_score":test_score} 
        #     print()
        #     print("evaluate result dict: ", evaluate_result)
        #     print()

            # summarize results
            print()
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            print()
            # for checking pickle file exist
            print("---pickle saving..")
            
            
            X = {}
            y = {}
            key_name_X = "X_"
            key_name_y = "y_"

            key_name_X += str(idx_time_unit) + "_" + str(idx_window_size) + "_" + str(idx_gap) + "_" + str(idx_margin_rate)
            key_name_y += str(idx_time_unit) + "_" + str(idx_window_size) + "_" + str(idx_gap) + "_" + str(idx_margin_rate)
            if (os.path.isfile(pickle_result_dir_path + \
                              MODEL + "_" + \
                              coin + "_" + \
                              str(idx_time_unit) + "_" + \
                              str(idx_window_size) + "_" + \
                              str(idx_gap) + "_" + \
                              str(idx_margin_rate) + \
                              "_result.pickle")) is not True:
                with open(pickle_result_dir_path + \
                          MODEL + "_" + \
                          coin + "_" + \
                          str(idx_time_unit) + "_" + \
                          str(idx_window_size) + "_" + \
                          str(idx_gap) + "_" + \
                          str(idx_margin_rate) + \
                          "_result.pickle", 'wb') as handle:
                    pickle.dump(evaluate_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    
            else:
                print("Already exist the file: ", pickle_result_dir_path + \
                                                  "_test_" + \
                                                  MODEL + "_" + \
                                                  "BTC" + "_" + \
                                                  str(idx_time_unit) + "_" + \
                                                  str(idx_window_size) + "_" + \
                                                  str(idx_gap) + "_" + \
                                                  str(idx_margin_rate) + \
                                                  "_result.pickle")

            
        #     for mean, stdev, param in zip(means, stds, params):
        #         print("%f (%f) with: %r" % (mean, stdev, param))
        #     print()
            key_name_X = "X_"
            key_name_y = "y_"


        #     return eval_result