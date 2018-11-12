
# coding: utf-8

# In[4]:


import pickle


# In[5]:


def Load_Dataset_X(dir_path, time_unit, window_size, gap, margin_rate):
    key_name_X = "X_" +                     str(time_unit) + "_" +                     str(window_size) + "_" +                     str(gap) + "_" +                     str(margin_rate)
                    

    with open(dir_path + key_name_X + ".pickle", 'rb') as handle:
        b_x = pickle.load(handle)
    return b_x
    
def Load_Dataset_y(dir_path, time_unit, window_size, gap, margin_rate):
    key_name_y = "y_" +                     str(time_unit) + "_" +                     str(window_size) + "_" +                     str(gap) + "_" +                     str(margin_rate)
                    

    with open(dir_path + key_name_y + ".pickle", 'rb') as handle:
        b_y = pickle.load(handle)
    return b_y


# In[6]:


dataset_dir_path_tuple_type = "./dataset_pickle_tuple_type/"

key_name_X = "X_10_10_1_0.1.pickle"
key_name_y = "y_10_10_1_0.1.pickle"
pickle_load_dir_path = dataset_dir_path_tuple_type
idx_time_unit = 10
idx_window_size = 10
idx_gap = 1
idx_margix_rate = 0.1

X = Load_Dataset_X(pickle_load_dir_path, idx_time_unit, idx_window_size, idx_gap, idx_margix_rate)
y = Load_Dataset_y(pickle_load_dir_path, idx_time_unit, idx_window_size, idx_gap, idx_margix_rate)


# In[7]:


X


# In[8]:


y


# In[9]:


X.shape


# In[10]:


type(X)


# In[11]:


X[-1]

