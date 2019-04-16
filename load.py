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

model = keras.models.load_model('MobileNet.h5')
