from torch.utils.data import Dataset
from sklearn import preprocessing

from utils import AudioUtils

class CovidCoughAudio(Dataset):
    def __init__(self, df, data_path) -> None:
        super().__init__()
        
        self.df = df
        self.data_path = data_path
        
        self.duration = 10.0
        self.sample_rate = 48000
        self.channels = 1
        
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(self.df['status'])
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]

        signal, sample_rate, info = AudioUtils.load_audio_file(row['uuid'])
        
        signal = AudioUtils.resample(signal, sample_rate, self.sample_rate)
        signal = AudioUtils.rechannel(signal, self.channels)

        signal = AudioUtils.resize(signal, self.duration, self.sample_rate)

        spectrogram = AudioUtils.spectrogram(signal, self.sample_rate)
        # TODO: Add augmentation
        
        label_id = self.label_encoder.transform([row['status']])[0]
        
        return spectrogram, label_id
