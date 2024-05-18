from PIL import Image
import numpy as np
import sys
import os
import cv2
from google.colab.patches import cv2_imshow
import csv
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop
from keras import regularizers
import matplotlib.pyplot as plt
%matplotlib inline

#can use Keras models as classifiers enabling easy experimentation and ombining the stregths of both frameworks
!pip install scikeras
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler

columnNames = list()
columnNames.append('label')

# This for loop iterates over a range of 2400
# Within each iteration of the loop, the variable pixel is assigned the string representation of the current iteration value 'i'
for i in range(2400):
    pixel = str(i)
    columnNames.append(pixel)

dataset = pd.DataFrame(columns = columnNames)

