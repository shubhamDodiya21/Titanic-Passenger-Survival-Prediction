from __future__ import absolute_import,division,print_function,unicode_literals
from tensorflow import feature_column
from six.moves import urllib
from os import system,name

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import os

def clear():
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('clear')

# Load the training and testing data
dfTrain = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv") # Testing data
dfEval = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/eval.csv") # Training data
y_train = dfTrain.pop('survived') # pop the survived tab from train.csv
y_eval = dfEval.pop('survived') # pop the survived tab from eval.csv

#convert all data in numeric manner
CATEGORICAL_COLUMNS = ['sex','n_siblings_spouses','parch','embark_town','class','deck','alone']

NUMERICAL_COLUMNS = ['age','fare']

feature_column = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dfTrain[feature_name].unique() # Export the unique from data
    feature_column.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name,vocabulary))
for feature_name in NUMERICAL_COLUMNS:
    feature_column.append(tf.feature_column.numeric_column(feature_name,dtype=tf.float32))

# Create the input fuction that convert the data in tf.data.dataset
def make_input_function(data_df,label_df,num_epochs = 10, shuffle = True,batch_size = 32):
    def input_function(): #Create the inner func
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df),label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function
train_input_fn = make_input_function(dfTrain,y_train)
eval_input_fn = make_input_function(dfEval,y_eval,num_epochs=1,shuffle=False)

#Train the modal
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_column)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

clear()

print(result['accuracy'])

result = list(linear_est.predict(eval_input_fn))
print('__'*25)
print(dfEval.loc[0])
print(f"Survived : {y_eval[0]}  (0 - Not survived , 1 - Survived)")
print(f"probabilities of survival : {result[0]['probabilities'][1]}")
print('__'*25)     

