# Credit Card Fraud Detection with Keras Autoencoder
# July 16, 2019
# Sung Kyu Lim
# Georgia Institute of Technology
# limsk@ece.gatech.edu


# import packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# global constants and hyper-parameters
RANDOM_SEED = 5234
MY_SPLIT = 0.25
DIM_ENCODER = 16
MY_EPOCH = 5
MY_BATCH = 200
MY_RECALL = 0.5


# directories
DB_DIR = "./database/"
OUT_DIR = "./output"
MODEL_DIR = "./model"
LOG_DIR = "./logs"


# create directories
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


    ####################
    # DATABASE SETTING #
    ####################


# load the database from a file and process it
def build_DB():

    # read the file
    raw_DB = pd.read_csv(DB_DIR + "creditcard.csv")

    # print some statistics of the raw DB
    print('\n== GENERAL DB INFO ==')
    print(raw_DB.info())
    print('\n== FIRST 5 DATA (RAW) ==')
    print(raw_DB.head())
    print('\n== DB STATISTICS (RAW) ==')
    print(raw_DB[['Amount', 'Class']].describe())

    print('\nTotal number of transactions', raw_DB.shape[0])
    print('Total number of fraud transactions:', raw_DB['Class'].sum())

    # exit()

    # drop the "Time" column (axis = 1)
    # then perform data centering of "Amount" column with z-score
    # we need to reshape the column to 2-dimension
    # to use fit_transform()
    scaler = StandardScaler()
    raw_DB = raw_DB.drop(['Time'], axis = 1)

    amount = raw_DB['Amount'].values
    amount = amount.reshape(-1, 1)
    raw_DB['Amount'] = scaler.fit_transform(amount)

    print('\n== FIRST 5 DATA (SCALED) ==')
    print(raw_DB.head())
    # exit()

    return raw_DB


# split the database to train and test sets
def split_DB(clean_DB):

    # python function from scikitlearn package
    # splits arrays or matrices into random train and test sets
    X_train, X_test = train_test_split(clean_DB, test_size = MY_SPLIT)
    

    # process train input set
    # remove all fraud data (Class = 1) from train data
    # then remove "Class" column from train data
    # all this is for anomaly detection
    print('\n== TRAIN AND TEST SET PROCESSING ==')
    print('Train input set shape (original):', X_train.shape)
    X_train = X_train[X_train.Class == 0]
    X_train = X_train.drop(['Class'], axis = 1)
    print('Train input set shape (processed):', X_train.shape)
    # exit()

    # process test input set
    # remove "Class" column and assign it to Y_test
    # we do NOT drop fraud transactions here
    # purpose is for confusion matrix and F1 score calculation
    print('Test input set shape (original):', X_test.shape)
    Y_test = X_test['Class']
    X_test = X_test.drop(['Class'], axis = 1)
    print('Test input set shape (processed):', X_test.shape)
    print('Test output set shape:', Y_test.shape)


    # the first column indicates which row in the excel file
    # the rows are shuffled during splitting
    print('\n== FIRST 5 TRAIN INPUT ==')
    print(X_train.head())
    print('\n== FIRST 5 TEST INPUT ==')
    print(X_test.head())
    print('\n== FIRST 5 TEST OUTPUT ==')
    print(Y_test.head())

    return X_train, X_test, Y_test


# Loading and processing data
clean_DB = build_DB()


# split the database to train and test sets
# note that we do NOT use training set label (Y_training)!!!
X_train, X_test, Y_test = split_DB(clean_DB)


    ###############################
    # MODEL BUILDING AND TRAINING #
    ###############################


# define encoder and decoder
def define_model():

    # 29 inputs are entering the encoder
    dim_input = X_train.shape[1]
    input = Input(shape = (dim_input,))
    print('\nEncoder input shape:', dim_input)


    # encoder: 29 -> 16 -> 8 neurons in a DNN
    encoder = Dense(DIM_ENCODER, activation = "tanh")(input)
    encoder = Dense(int(DIM_ENCODER / 2), activation = "relu")(encoder)


    # decoder: 8 -> 29 neurons in a DNN
    decoder = Dense(int(DIM_ENCODER / 2), activation = 'tanh')(encoder)
    decoder = Dense(dim_input, activation = 'relu')(decoder)


    # combining encoder and decoder to form autoencoder
    autoencoder = Model(inputs = input, outputs = decoder)
    autoencoder.summary()

    return autoencoder


# plot loss curve using the history collected during training
def plot_loss_curves(history):

    # add title and determine the plot size 
    fig = plt.figure(num = "Loss Curve")
    fig.set_size_inches(12, 6)
    

    # adding axis labels and legend
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Loss By Epoch')
    plt.ylabel('Loss')
    plt.xlabel('Epoch Num')
    plt.legend(['X_train', 'X_test'], loc = 'upper right');
    plt.grid(True, alpha = 0.25)
    plt.tight_layout()


    # save the plot and close it
    image_name = 'chap5.png'
    fig.savefig(os.path.join(OUT_DIR, image_name))
    plt.clf()


# train the autoencoder
def train_model():
    model.compile(optimizer = 'adam', loss = 'mse', metrics = ['acc'])

    # keras package to save the model after every epoch
    checkpoint = ModelCheckpoint(filepath = os.path.join(MODEL_DIR, "chap5.h5"),
                                    verbose = 0,
                                    save_best_only = True)


    # TensorBoard is a visualization tool provided with TensorFlow
    # tensorboard --logdir=logs
    # localhost:6006
    log_tensorboard = TensorBoard(log_dir='./logs',
                                    histogram_freq = 1,
                                    write_graph = True,
                                    write_images = True)


    # training with keras callback utilities
    # we can use callbacks to get a view on internal states and statistics 
    # of the model during training
    # we can pass a list of callbacks to the .fit() 
    # note that we use X_train for both the input and the output
    history = model.fit(X_train, X_train,
                        epochs = MY_EPOCH,
                        batch_size = MY_BATCH,
                        shuffle = True,
                        validation_data = (X_test, X_test),
                        verbose = 1,
                        callbacks = [checkpoint, log_tensorboard]).history


    # training done. plotting loss curves
    plot_loss_curves(history)
    return history


# define and train the model
model = define_model()
history = train_model()


    ####################
    # MODEL EVALUATION #
    ####################


# compute reconstruction error using the autoencoder
# we use test set and its prediction 
# then calculate mean square error between the two
def recon_error():

    # use autoencoder to obtain prediction from test set
    pred = model.predict(X_test)
    print('\n== RECONSTRUCTION ERROR CALCULATION INFO ==')
    print("Test set shape:", X_test.shape)   
    print("Prediction shape:", pred.shape)       
    print('Showing prediction data')
    print(pred)

    # exit()

    # MSE calculation
    # axis = 0 means along the column, axis = 1 along the row.
    # we create a new pandas dataframe to store the error and label
    mse = np.mean(np.power(X_test - pred, 2), axis = 1)
    recon_error = pd.DataFrame({'recon_error': mse, 'true_class': Y_test})
    print("\nMean square error shape:", mse.shape)    
    print("Reconstruction error shape:", recon_error.shape)
    print('Showing error data')
    print(recon_error)

    # exit()

    # print stats on fraud transactions
    # we see the average is very high
    fraud = recon_error[(recon_error['true_class'] == 1)]
    print('\n== RECONSTRUCTION ERROR STATS ==')    
    print('There are', len(fraud), 'fraud data in the test set.')
    print('Showing fraud data in the test set:')
    print(fraud)   
    
    # exit()

    # print stats on reconstruction error
    # we see the average is very low
    test = recon_error[(recon_error['true_class'] == 0)]
    print('\nOverall reconstruction error data stats:')
    print(test.describe())

    return recon_error


# first we seek the threshold that gives at least 0.8 (= MY_RECALL) recall
# next we use it to build confusion matrix and calculate F1 score
def get_confusion_matrix(error):
    
    # we use precision_recall_curve() from scikit-learn package
    # we use recall and threshold and ignore precision
    _ , recall, threshold = precision_recall_curve(error.true_class, 
            error.recon_error)

    # print recall and threshold details
    print('\n== RECALL AND THRESHOLD INFO ==')
    print('There are', len(recall), 'recall values.')
    print('There are', len(threshold), 'threshold values.')

    print('\nRecall data:')
    print(recall)
    print('\nThreshold data:')
    print(threshold)


    # searching for the threshold that gives at least 0.8 (= MY_RECALL) recall
    # recall value is decreasing while threshold is increasing
    where = np.where(recall <= MY_RECALL)
    idx = where[0][0]
    th_val = threshold[idx]
    print('\nIndex:', idx)
    print('Threshold:', th_val)
    print('Recall:', recall[idx])


    # build binary prediction array
    # we use the threshold calculated above
    # if the reconstruction error value is greater than the threshold
    # we mark it as fraud
    # pred: 0 means not fraud, 1 is fraud
    pred = [1 if val > th_val else 0 for val in error.recon_error.values]
    c_matrix = confusion_matrix(error.true_class, pred)
    print('\n== PREDICTION VS. GROUND TRUTH INFO ==')
    print('Number of test data:', len(pred))
    print('Number of non-fraud data:', len(pred) - sum(error.true_class))
    print('Number of fraud data:', sum(error.true_class))


    # show confusion matrix
    # positive: not fraud, negative: fraud
    print('\n== CONFUSION MATRIX ==')
    print(c_matrix)
    print("\nF1 score:", f1_score(error.true_class, pred, average = 'micro'))


# load the model after training
# this is not necessary actually
load_model(os.path.join(MODEL_DIR, "chap5.h5"))


# calculate reconstruction error
error = recon_error()


# seek the threshold that gives at least 0.8 (= MY_RECALL) recall
# use the threshold to build confusion matrix and calculate F1 score
get_confusion_matrix(error)


# 결과 해석
# threshold값으로 나온 값이 recall 0.8을 보장하는 값이다.
