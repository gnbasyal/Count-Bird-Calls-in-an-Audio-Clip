from preprocess.preprocess import Preprocess
from configs.model_config import ModelConfig
import tensorflow as tf
import numpy as np
from itertools import groupby

def getModel():
    model_path = fr"{ModelConfig.model_dump_path}\capuchin_model.h5"
    model = tf.keras.models.load_model(model_path)
    return model

def getCount(file_path):
    spectrograms = Preprocess.mp3_to_spectrograms(file_path)
    model = getModel()
    y_hat = model.predict(np.array(spectrograms))
    y_hat = list(map(int,(y_hat>0.5)))
    # print(y_hat)

    y_hat = [key for key,group in groupby(y_hat)]
    count = sum(y_hat)
    
    return count


if __name__ == "__main__":
    file_path = r'D:\Ganesh\Count Bird Calls\data\Forest Recordings\recording_15.mp3'
    print(getCount(file_path))