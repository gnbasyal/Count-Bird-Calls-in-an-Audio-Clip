from preprocess.preprocess import Preprocess
from matplotlib import pyplot as plt
import tensorflow as tf
from configs.project_config import ProjectConfig
from datetime import datetime
import os

def visualize_waveform(filename, color = 'red', show = True, save = False):
    wave = Preprocess.load_wav_16k_mono(filename)
    plt.plot(wave, color)
    if show:
        plt.show()
    if save:
        filename = filename.split('\\')[-1]
        save_file_name = fr"{filename}~{str(datetime.now()).replace(' ','_').replace(':','.')}.png"
        save_path = fr"{ProjectConfig.wav_dump_path}\{save_file_name}"
        os.makedirs(ProjectConfig.wav_dump_path, exist_ok=True)
        plt.savefig(save_path)

def visualize_spectrogram(filename, show = True, save = False):
    spectrogram, label = Preprocess.get_spectrogram(filename)
    plt.figure(figsize=(30,20))
    plt.imshow(tf.transpose(spectrogram)[0])
    plt.title(label=label)
    if show:
        plt.show()
    if save:
        filename = filename.split('\\')[-1]
        save_file_name = fr"{filename}~{str(datetime.now()).replace(' ','_').replace(':','.')}.png"
        save_path = fr"{ProjectConfig.spec_dump_path}\{save_file_name}"
        os.makedirs(ProjectConfig.spec_dump_path, exist_ok=True)
        plt.savefig(save_path)

if __name__ == "__main__":
    # print(load_wav_16k_mono)
    visualize_waveform(r'D:\Ganesh\Count Bird Calls\data\Parsed_Capuchinbird_Clips\XC22397-4.wav',show=False, save=True)
    visualize_spectrogram(r'D:\Ganesh\Count Bird Calls\data\Parsed_Capuchinbird_Clips\XC22397-4.wav',show=False, save=True)