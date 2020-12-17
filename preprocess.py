import numpy as np
import matplotlib.pyplot as plt
import os
import json
import math
import librosa
import librosa.display
import cv2
from utils import split_and_save


def load_music_audios(path):
    MUSIC = path + 'genres/'

    genres = os.listdir(MUSIC)
    musics = []
    for genre in genres:
        if genre == ".DS_Store":
            continue
        music_path = MUSIC + genre

        for name in os.listdir(music_path):
            if name == ".DS_Store":
                continue
            musics.append(os.path.join(music_path, name))

    return musics

def draw_spectrogram(music_file):
    music, sample_rate = librosa.load(music_file)
    D = np.abs(librosa.stft(music, n_fft=2048, hop_length=512))
    librosa.display.specshow(D, sr=sample_rate, x_axis="time", y_axis="linear")
    plt.colorbar()
    # plt.show()

    plt.savefig("spec.png", bbox_inches='tight')
    plt.close()

    music, sample_rate = librosa.load(music_file)
    n_mels=128
    melSpec = librosa.feature.melspectrogram(y=music, sr=sample_rate, n_mels=n_mels)
    melSpec_dB = librosa.power_to_db(melSpec, ref=np.max)

    librosa.display.specshow(
            melSpec_dB, sr=sample_rate, fmax=8000, hop_length=512, x_axis="time", y_axis="linear")
    plt.colorbar()

    plt.savefig("mel_spec.png", bbox_inches='tight')
    plt.close()

def generate_mel_spectrograms_for_music(music_file, output_path, num_segments=6, n_mels=128, default_size=True, size=(1.28, 1.28)):
    duration = 30
    music, sample_rate = librosa.load(music_file)
    samples_per_segment = int(sample_rate * duration / num_segments)

    for i in range(num_segments):
        start = i * samples_per_segment
        end = start + samples_per_segment

        melSpec = librosa.feature.melspectrogram(
            y=music[start:end+1], sr=sample_rate, n_mels=n_mels)
        melSpec_dB = librosa.power_to_db(melSpec, ref=np.max)

        if default_size:
            plt.figure(frameon=False)
        else:
            plt.figure(frameon=False, figsize=size, dpi=100)
        librosa.display.specshow(
            melSpec_dB, sr=sample_rate, fmax=8000, hop_length=512)
        output_name = '{}_segment_{}.png'.format(
            "_".join(music_file.split("/")[-1].split(".")[:-1]), i)
        plt.savefig(os.path.join(output_path, output_name))
        plt.close()


def generate_mel_spectrograms(music_files, images_path, name, num_segments=6, n_mels=128, default_size=True, size=(1.28, 1.28)):
    images_path = os.path.join(images_path, name)
    if not os.path.exists(images_path):
        os.mkdir(images_path)

    for music_file in music_files:
        genre = music_file.split("/")[-2]
        file_path = os.path.join(images_path, genre)
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        generate_mel_spectrograms_for_music(
            music_file, file_path, num_segments, n_mels, default_size, size)


processed_path_base = "processed_images"

if not os.path.exists(processed_path_base):
    os.mkdir(processed_path_base)

path = "./"
music_files = load_music_audios(path)

print("Drawing Spectrogram and Mel-Spectrogram...")
# draw and save Figure 1 and Figure 2 in the paper
draw_spectrogram(music_files[0])


print("Generating music audio image data...")
names=["128_128_6seg", "128_128_10seg", "128_128_15seg"]
# generate music audio image data used in the paper
# 128 x 128, 6 segments
generate_mel_spectrograms(music_files, processed_path_base, "128_128_6seg",
                          num_segments=6, n_mels=128, default_size=False, size=(1.28, 1.28))
print("128 x 128, 6 segments is done.")


# 128 x 128, 10 segments
generate_mel_spectrograms(music_files, processed_path_base, "128_128_10seg",
                          num_segments=10, n_mels=128, default_size=False, size=(1.28, 1.28))
print("128 x 128, 10 segments is done.")

# 128 x 128, 15 segments
generate_mel_spectrograms(music_files, processed_path_base, "128_128_15seg",
                          num_segments=15, n_mels=128, default_size=False, size=(1.28, 1.28))
print("128 x 128, 15 segments is done.")


print("Spliting image data for training and testing...")
data_path = "saved_dataset/"
if not os.path.exists(data_path):
        os.mkdir(data_path)

for name in names:
    split_and_save(data_path, name)
    print(name + " is done.")