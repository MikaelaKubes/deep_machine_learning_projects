#!/usr/bin/env python

import numpy as np 
import pandas as pd 
import sys
import argparse
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

def run_classifier(input_file):
    df = pd.read_csv(input_file)
    X = df.iloc[:, 0:4].values
    y = df.iloc[: , 4].values

    # Encoding categorical labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    y = np_utils.to_categorical(y)

    # Splitting into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Defining model
    model = Sequential()
    model.add(Dense(8, activation='relu', input_dim=4))
    model.add(Dense(8))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    model.fit(X_train, y_train, batch_size = 5, epochs = 100,
        validation_data=(X_test, y_test))


def main(argv):
    p = argparse.ArgumentParser(
        description="Multiclass classification between the " +
        "iris classes")

    p.add_argument("-i", "--input-file",
                   help="The csv input file that contains " +
                   "iris dataset",
                   required=True)

    args = vars(p.parse_args())

    input_file = args.get("input_file")

    run_classifier(input_file)

if __name__ == "__main__":
   main(sys.argv[1:])