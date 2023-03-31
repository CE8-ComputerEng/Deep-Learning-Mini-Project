import numpy as np
import matplotlib.pyplot as plt
import torch

import warnings
warnings.filterwarnings('ignore')

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, filename=None):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)

    plt.figure(figsize=(6, 6))
    
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    plt.xticks(np.arange(len(classes)), classes)
    plt.yticks(np.arange(len(classes)), classes)
    
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
    
    if filename:
        plt.savefig(filename)
    
    plt.show()
    

def plot_label_distribution(value_counts, title='Distribution of samples in dataset', filename=None):
    plt.figure(figsize=(6, 4))
    
    plt.bar(value_counts.index, value_counts.values)
    
    plt.title(title)
    plt.xlabel('Status of sample')
    plt.ylabel('Number of samples')
    
    if filename:
        plt.savefig(filename)
    
    plt.show()


def plot_audio_waveform(signal, sample_rate, title='Audio waveform', filename=None):
    if type(signal) == torch.Tensor:
        signal = torch.clone(signal)
        signal = signal.detach().cpu().numpy()
    
    if len(signal.shape) > 1:
        signal = signal[0]
    
    NUM_TICKS = 10
    
    plt.figure(figsize=(10, 5))

    plt.plot(signal)

    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.xticks(np.linspace(0, len(signal), NUM_TICKS), [f'{t:.2f}' for t in np.linspace(0, len(signal) / sample_rate, NUM_TICKS)])
    plt.grid()

    if filename:
        plt.savefig(filename)
    
    plt.show()
    

def plot_audio_spectogram(spectrogram, length, title='Mel-Spectrogram (dB)', filename=None):
    if type(spectrogram) == torch.Tensor:
        spectrogram = torch.clone(spectrogram)
        spectrogram = spectrogram.detach().cpu().numpy()
    
    if len(spectrogram.shape) > 2:
        spectrogram = spectrogram[0]
    
    plt.figure(figsize=(15, 5))

    im = plt.imshow(spectrogram, aspect='auto', origin='lower', extent=[0, length, 0, spectrogram.shape[0]])
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Mel bins')
    
    if filename:
        plt.savefig(filename)