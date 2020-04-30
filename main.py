import sys
import os, shutil

import re

import time
import subprocess

from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd

import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold

from sklearn.utils import class_weight

from sklearn.decomposition import PCA

import tensorflow as tf

mpl.rcParams['figure.dpi']= 300

RANDOM_STATE = 42

list_dataset_filename = ['dataset_astral-scopedom-seqres-gd-sel-gs-bib-40-2.0.7.csv',
                            'dataset_astral-scopedom-seqres-gd-sel-gs-bib-95-2.0.7.csv']

dataset_filename = list_dataset_filename[1]
dataset_name = os.path.splitext(dataset_filename)[0]

dataset = pd.read_csv(dataset_filename)

dataset = dataset.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


dataset_nullcheck = dataset.isnull().any()

null_columns = dataset_nullcheck[dataset_nullcheck == True]

print("Number of columns that contain null values :", len(null_columns))
print("Names of columns that contain null values :", null_columns)

categorical_cols = [ col for col in dataset.columns if dataset[col].dtype == 'object' ]

def class_percentage_occurance(dataset):
    return (dataset['class'].value_counts() * 100 / dataset['class'].size).to_frame(name='Percentage Occurance').T

unwanted_cols = ['sid', 'sequence', 'folds', 'superfamilies', 'families']

dataset = dataset.drop(unwanted_cols, axis=1)

def trim_dataset(dataset, classes):
    """
    Reduces the dataset by keeping the classes specified, and dropping the rest
    """
    # using .loc would later on avoid the SettingWithCopyWarning that would appear with LabelEncoding
    # basically it helps to avoids returning a copy of the DataFrame
    return dataset[dataset['class'].isin(classes)].loc[:, dataset.columns]

wanted_classes = ['a', 'b', 'c', 'd']

dataset = trim_dataset(dataset, wanted_classes)

def label_encode_dataset(dataset, classes):
    """
    Label Encodes the specified classes in the Dataset
    The classes parameter can be `list` or `str`
    
    If the type(classes) is `list`
    The dataset is first reduced to only match the classes specified
    Then it is Label Encoded
    
    If the type(classes) is `str`
    We label encode the classes by `class` and `not class` (1, 0 respectively)
    """
    dataset_clone = dataset.copy(deep=True)
    
    if type(classes) is list:
        dataset_clone = trim_dataset(dataset_clone, classes)
        
        label_encoder = LabelEncoder()
        dataset_clone['class'] = label_encoder.fit_transform(dataset_clone['class'])
    elif type(classes) is str:
        dataset_clone['class'] = dataset_clone['class'].map(lambda x: int(classes == x))
    else:
        raise Exception('Unsupported Argument Type \'classes\'; Should be \'list\' or \'str\'')
    
    return dataset_clone

labelencoded_df = label_encode_dataset(dataset, wanted_classes)

X = labelencoded_df.drop(['class'], axis=1)
y = labelencoded_df['class']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

class_weights = dict(enumerate(
                    class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)))

def getXData(X_df, data_shape):
    return np.apply_along_axis(lambda x: x.reshape(data_shape), axis=1, arr=X_df.values)

def getYData(y_sequence):
    return y_sequence.values

data_scale_factor = getXData(X, (len(X.columns), )).max()
rescale_factor = 1./data_scale_factor

def createClusterMatrix(X_row):
    len_X_row = len(X_row)
    
    matt = np.zeros(shape=(len_X_row, len_X_row))
    
    for i in range(len_X_row):
        if (X_row[i] > 0):
            matt[i, i] = X_row[i]
            
    return matt

def getClusterMatrix(X_df):
    X_data = getXData(X_df, (len(X_df.columns), ))
    
    return np.apply_along_axis(createClusterMatrix, axis=1, arr=X_data)

img_shape_rgba = (238, 238, 4)
img_shape_grayscale = (476, 476, 1)

def getImageData(X_df, img_shape):
    cluster_matts = getClusterMatrix(X_df)
    
    return cluster_matts.reshape(((len(cluster_matts), ) + img_shape))

def printImage(img_data, dpi=50):
    img_shape = img_data.shape
    
    plt.figure(figsize=(img_shape[0] / 10, img_shape[1] / 10), dpi=dpi)
    plt.axis('off')
    plt.imshow(img_data, cmap='gray')
    plt.show()

# printImage(getImageData(X.sample(1), img_shape_rgba)[0] * rescale_factor, dpi=10)

# printImage(getImageData(X.sample(1), img_shape_grayscale[:-1])[0] * rescale_factor, dpi=5)

img_shape = img_shape_rgba

batch_size = 128

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=rescale_factor)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=rescale_factor)

train_generator = train_datagen.flow(getImageData(X_train, img_shape), y=getYData(y_train),
                                        batch_size=batch_size, shuffle=True)

valid_generator = valid_datagen.flow(getImageData(X_valid, img_shape), y=getYData(y_valid),
                                        batch_size=batch_size, shuffle=True)

convnet_base = tf.keras.applications.VGG16(input_tensor=tf.keras.Input(shape=img_shape),
                                                include_top=False, weights=None,
                                                    pooling='avg')

convnet_base.trainable = True

fine_tune_at = 0

for layer in convnet_base.layers[:fine_tune_at]:
    layer.trainable =  False

def make_model(learning_rate=0.00001):
    model = tf.keras.Sequential()
    
    model.add(convnet_base)
    
    classification_layer = tf.keras.layers.Dense(len(wanted_classes),
                                                    activation='softmax')
    model.add(classification_layer)
    
    compile_model(model, learning_rate=learning_rate)
    
    return model

def compile_model(model, learning_rate=0.00001):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.),
                              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model = make_model(learning_rate=0.0001)

model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

checkpoint = tf.keras.callbacks.ModelCheckpoint('checkpoint.h5', monitor='val_accuracy', 
                                                    verbose=1, save_best_only=True, period=1,
                                                        save_weights_only=False, mode='auto')

fine_tune_epochs = 20

fine_tune_history = model.fit(train_generator, epochs=fine_tune_epochs,
                        validation_data=valid_generator, shuffle=True,
                           callbacks=[checkpoint, early_stopping])
