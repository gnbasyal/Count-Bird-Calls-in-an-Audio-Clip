from configs.dataset_config import DatasetConfig
from configs.model_config import ModelConfig
from configs.project_config import ProjectConfig
import os
from preprocess.preprocess import Preprocess
import numpy as np
import random
from matplotlib import pyplot as plt
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.metrics import Recall, Precision

def getFilepathsAndLabels():
    POS = DatasetConfig.positive_data_path
    NEG = DatasetConfig.negative_data_path
    # filepaths = [p.path for p in os.scandir(POS)] + [p.path for p in os.scandir(NEG)]
    pos_paths = [p.path for p in os.scandir(POS)]
    neg_paths = [p.path for p in os.scandir(NEG)]
    filepaths = pos_paths+neg_paths
    labels = [1]*len(pos_paths) + [0]*len(neg_paths)
    print(f'Found {len(pos_paths)} positive samples and {len(neg_paths)} negative samples.')
    return filepaths, labels

def getSpectrogramsAndLabels(filepaths, labels):
    specs_and_labels=list(map(Preprocess.get_spectrogram, filepaths, labels))
    print(f'Generated spectrograms for data samples.')
    return specs_and_labels

def getTrainAndTestData(specs_and_labels):
    total_length = len(specs_and_labels)
    shuffle_seed = 1
    random.Random(shuffle_seed).shuffle(specs_and_labels)
    X_full = [s[0] for s in specs_and_labels]

    # Resize spectrograms to be compatible with model
    X_full = list(map(lambda x: tf.image.resize(x, [350,50]), X_full))

    Y_full = [s[1] for s in specs_and_labels]
    X_train = np.array(X_full[:int(total_length*DatasetConfig.train_test_split)])
    Y_train = np.array(Y_full[:int(total_length*DatasetConfig.train_test_split)])
    X_test = np.array(X_full[int(total_length*DatasetConfig.train_test_split):])
    Y_test = np.array(Y_full[int(total_length*DatasetConfig.train_test_split):])
    print(f'Separated training and test data.')
    print(f'{len(X_train)=}, {len(Y_train)=}, {len(X_test)=}, {len(Y_test)=}')
    return X_train, Y_train, X_test, Y_test 

def getModel():
    model = Sequential()
    model.add(Conv2D(16, (3,3), activation='relu', input_shape=ModelConfig.input_image_shape,))
    model.add(Conv2D(16, (3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('Adam', loss='BinaryCrossentropy', metrics=[Recall(),Precision()])
    model.summary()
    return model

def plotMetrics(hist, path=ProjectConfig.plot_dump_path):

    os.makedirs(path, exist_ok=True)

    # Plot Loss
    plt.title('Loss')
    plt.plot(hist.history['loss'], 'r')
    plt.plot(hist.history['val_loss'], 'b')
    plt.savefig(fr"{path}\Loss_{(str(datetime.now())).replace(':','.')}.png")
    

    # Plot Precision
    plt.clf()
    plt.title('Precision')
    plt.plot(hist.history['precision'], 'r')
    plt.plot(hist.history['val_precision'], 'b')
    plt.savefig(fr"{path}\Precision_{(str(datetime.now())).replace(':','.')}.png")

    # Plot Recall
    plt.clf()
    plt.title('Recall')
    plt.plot(hist.history['recall'], 'r')
    plt.plot(hist.history['val_recall'], 'b')
    plt.savefig(fr"{path}\Recall_{(str(datetime.now())).replace(':','.')}.png")

def trainModel(model, X_train, Y_train, X_test, Y_test):
    hist = model.fit(x=X_train, y=Y_train, epochs=10, validation_data=(X_test,Y_test))

    plotMetrics(hist)

    model.save(fr"{ModelConfig.model_dump_path}\capuchin_model.h5")

def makePredictions(path):
    model = tf.keras.models.load_model(fr"{ModelConfig.model_dump_path}\capuchin_model.h5")
    spec, _ = Preprocess.get_spectrogram(path)
    spec = tf.image.resize(spec, ModelConfig.input_image_shape[:2])
    prediction = model.predict(np.array([spec]))
    return prediction[0][0]
    

if __name__ == "__main__":
    # filepaths, labels = getFilepathsAndLabels()
    # specs_and_labels = getSpectrogramsAndLabels(filepaths, labels)
    # X_train, Y_train, X_test, Y_test = getTrainAndTestData(specs_and_labels)
    # model = getModel()
    # trainModel(model, X_train, Y_train, X_test, Y_test)
    print(makePredictions(r"D:\Ganesh\Count Bird Calls\data\Parsed_Not_Capuchinbird_Clips\crickets-chirping-crickets-sound-27.wav"))