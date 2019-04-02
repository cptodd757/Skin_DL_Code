import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import sys
import shutil
from sklearn.model_selection import train_test_split
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Input

datagen = ImageDataGenerator(preprocessing_function = keras.applications.mobilenet.preprocess_input,
                                    rotation_range = 180,
                                    width_shift_range = 0.1,
                                    height_shift_range = 0.1,
                                    zoom_range = 0.1,
                                    horizontal_flip = True,
                                    vertical_flip = True,
                                    #brightness_range = (0.9,1.1),
                                    fill_mode = 'nearest')

test_batches = datagen.flow_from_directory('test_dir', target_size = (64, 64), batch_size  = 1, shuffle=False)

mobile = keras.applications.mobilenet.MobileNet(input_tensor = Input(shape = (64, 64, 3)))

# Fine tune the model for the last few layers
x = mobile.layers[-6].output
x = Dropout(0.25)(x)
predictions = Dense(7, activation='softmax')(x)
model = Model(inputs = mobile.input, outputs = predictions)
model.summary()

def top_3_accuracy(y_true, y_pred):
  return top_k_categorical_accuracy(y_true, y_pred, k=3)
def top_2_accuracy(y_true, y_pred):
  return top_k_categorical_accuracy(y_true, y_pred, k=2)

model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy])



model.load_weights("MobileNet.h5")
val_loss, val_cat_acc, val_top_2_acc, val_top_3_acc = model.evaluate_generator(test_batches, steps=len(os.listdir('test_dir'))
print('val_loss:', val_loss)
print('val_cat_acc:', val_cat_acc)
print('val_top_2_acc:', val_top_2_acc)
print('val_top_3_acc:', val_top_3_acc)
