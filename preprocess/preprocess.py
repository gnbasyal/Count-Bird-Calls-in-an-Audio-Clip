from configs.dataset_config import DatasetConfig
import librosa as lb
import tensorflow as tf
import numpy as np
# print(DatasetConfig.positive_data_path)

class Preprocess:
    def load_wav_16k_mono(filename):
        wave, sr = lb.load(filename, sr = 16000, mono = True)
        return wave

    def get_spectrogram(file_path, label=None): 
        wav = Preprocess.load_wav_16k_mono(file_path)
        wav = wav[:48000]
        zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
        wav = tf.concat([zero_padding, wav],0)
        spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.expand_dims(spectrogram, axis=2)
        # spectrogram = tf.image.resize(spectrogram, [350,50])
        return spectrogram, label

    def split_waveform(waveform, frame_size=3, sample_rate=16000):
        # Calculate the number of samples in each frame
        frame_length = frame_size * sample_rate

        # Split the waveform into frames
        num_frames = len(waveform) // frame_length
        waveform_frames = np.array_split(waveform[:num_frames * frame_length], num_frames)

        return waveform_frames

    def wave_to_spectrogram(wav):
        spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.expand_dims(spectrogram, axis=2)
        spectrogram = tf.image.resize(spectrogram, [350,50])
        return spectrogram

    def mp3_to_spectrograms(filename):
        wave = Preprocess.load_wav_16k_mono(filename)
        splits = Preprocess.split_waveform(wave, frame_size=3, sample_rate=16000)
        spectrograms = list(map(Preprocess.wave_to_spectrogram, splits))
        return spectrograms