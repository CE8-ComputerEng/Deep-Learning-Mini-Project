import pandas as pd
import os
import vlc

data_folder = input("Insert path to data folder\n")

dataframe = pd.read_csv(os.path.join(data_folder, 'metadata_compiled.csv'))

healthy_samples = dataframe[dataframe['status'] == 'healthy']
covid_samples = dataframe[dataframe['status'] == 'COVID-19']
symptomatic_samples = dataframe[dataframe['status'] == 'symptomatic']

print("\nPlaying healthy sample")
if os.path.exists(os.path.join(data_folder, healthy_samples.iloc[0]['uuid'])+".webm"):
    vlc_player = vlc.MediaPlayer(os.path.join(data_folder, healthy_samples.iloc[0]['uuid'])+".webm")
else:
    vlc_player = vlc.MediaPlayer(os.path.join(data_folder, healthy_samples.iloc[0]['uuid'])+".ogg")
vlc_player.play()
input("Press Enter to continue...")

print("\nPlaying covid sample")
if os.path.exists(os.path.join(data_folder, covid_samples.iloc[0]['uuid'])+".webm"):
    vlc_player = vlc.MediaPlayer(os.path.join(data_folder, covid_samples.iloc[0]['uuid'])+".webm")
else:
    vlc_player = vlc.MediaPlayer(os.path.join(data_folder, covid_samples.iloc[0]['uuid'])+".ogg")
vlc_player.play()
input("Press Enter to continue...")

print("\nPlaying symptomatic sample")
if os.path.exists(os.path.join(data_folder, symptomatic_samples.iloc[0]['uuid'])+".webm"):
    vlc_player = vlc.MediaPlayer(os.path.join(data_folder, symptomatic_samples.iloc[0]['uuid'])+".webm")
else:
    vlc_player = vlc.MediaPlayer(os.path.join(data_folder, symptomatic_samples.iloc[0]['uuid'])+".ogg")
vlc_player.play()
input("Press Enter to continue...")
