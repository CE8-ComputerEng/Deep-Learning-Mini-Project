import glob
import random
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import pytorch_lightning as pl
from sklearn import preprocessing


class AudioUtils():
    @staticmethod
    def get_audio_file(uuid, data_path):
        results = glob.glob(data_path + uuid + '.*')
        for result in results:
            ext = result.split('.')[1]
            if ext in ['ogg', 'webm']:
                return result
        else:
            raise Exception('No audio file found for uuid: {}'.format(uuid))
    
    
    @staticmethod
    def load_audio_file(uuid, data_path):
        audio_file = AudioUtils.get_audio_file(uuid, data_path)
        signal, sample_rate = torchaudio.load(audio_file)
        info = torchaudio.info(audio_file)
        
        return signal, sample_rate, info
    
    
    @staticmethod
    def resample(signal, sample_rate, target_sample_rate):
        if sample_rate == target_sample_rate:
            return signal
        
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        signal = resampler(signal)
        
        return signal
    
    
    @staticmethod
    def rechannel(signal, channels):
        if signal.shape[0] == channels:
            return signal
        
        if signal.shape[0] == 1 and channels == 2:
            return signal.repeat(2, 1)
        
        if signal.shape[0] == 2 and channels == 1:
            return signal.mean(dim=0, keepdim=True)
        
        raise Exception('Cannot convert from {} channels to {} channels'.format(signal.shape[0], channels))
    
    
    @staticmethod
    def resize(signal, duration, sample_rate):
        new_length = int(duration * sample_rate)
        singal_length = signal.shape[1]
        
        if singal_length == new_length:
            return signal
        elif singal_length > new_length:
            return signal[:, :new_length]
        elif singal_length < new_length:
            pad_begin_length = random.randint(0, new_length - singal_length)
            pad_end_length = new_length - singal_length - pad_begin_length
            
            pad_begin = torch.zeros((signal.shape[0], pad_begin_length))
            pad_end = torch.zeros((signal.shape[0], pad_end_length))
            
            signal = torch.cat((pad_begin, signal, pad_end), dim=1)
            
            return signal
    
    
    @staticmethod
    def get_spectrogram(signal, sample_rate, n_mels=64, n_fft=1024, top_db=80):
        spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft)(signal)
        spectrogram = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(spectrogram)
        
        return spectrogram


class CoughDataset(Dataset):
    def __init__(self, 
                 df, 
                 data_path,
                 duration=10.0,
                 sample_rate=48000,
                 channels=1,
                 n_mels=64,
                 n_fft=1024, 
                 top_db=80):
        super().__init__()
        
        self.df = df
        self.data_path = data_path
        
        self.duration = duration
        self.sample_rate = sample_rate
        self.channels = channels
        
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.top_db = top_db
        
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(self.df['status'])
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]

        signal, sample_rate, info = AudioUtils.load_audio_file(row['uuid'], self.data_path)
        
        signal = AudioUtils.resample(signal, sample_rate, self.sample_rate)
        signal = AudioUtils.rechannel(signal, self.channels)

        signal = AudioUtils.resize(signal, self.duration, self.sample_rate)

        spectrogram = AudioUtils.get_spectrogram(signal, self.sample_rate, self.n_mels, self.n_fft, self.top_db)
        # TODO: Add augmentation
        
        label_id = self.label_encoder.transform([row['status']])[0]
        
        return spectrogram, label_id