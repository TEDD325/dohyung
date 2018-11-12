from local_function import *
from start_model import *
from create_LSTM_model import *

def start(machine, 
          _GPU,
          n_jobs, 
          MODEL, 
          idx_time_unit, 
          idx_window_size, 
          idx_gap, 
          idx_margin_rate, 
          cv,
          dataset_scale,
          epochs):
    
    '''
        [ATTENTION] In create_model METHOD part, need to set appropriate about GPU
        
        LINK01 -> GPU OFF
        MSI -> GPU OFF
        SLAVE04 -> GPU ON
        SLAVE05 -> GPU ON
    ''' 
    

    start_time = time.time()
    evaluate_result = Start_Model(pickle_load_dir_path = "../_dataset/RNN_coin/",  
                pickle_result_dir_path = "./evaluate_result/", 
                idx_time_unit=idx_time_unit,
                idx_window_size=idx_window_size, 
                idx_gap=idx_gap, 
                idx_margin_rate=idx_margin_rate, 
                epochs=epochs, 
                MODEL=MODEL, 
                _GPU=_GPU,
                n_jobs=n_jobs,
                cv=cv,
                dataset_scale=dataset_scale,
                machine=machine)
    print("DEBUG")
    end_time = time.time()
    print()
    print("TIME: ", end_time-start_time)
    return evaluate_result