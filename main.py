import numpy as np
import pandas as pd

from net import *
from helpers import load_data

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X_train, y_train, X_test, y_test = load_data()
print(X_train)
X_train = X_train.reshape(X_train.shape[0], -1).T
X_test = X_test.reshape(X_test.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
X_train = X_train/255.
X_test = X_test/255.
