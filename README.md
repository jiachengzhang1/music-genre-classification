# music-genre-classification

## How to run experiments on Google Colab?

Upload the project folder to Google Drive, and open `MusicGenreClassification.ipynb` with Colab.

change `FOLDERNAME` to where `CSC583_FinalProject` folder is located in the Google Drive.

You can make a shortcut or copy the saved_dataset folder to the project folder, here's the link,
https://drive.google.com/drive/folders/1gB3JV7wtXPOL1nndgedToCSiXuZcwT5B?usp=sharing

Or, you could create a folder called `saved_dataset` in the project folder, and upload `.pkl` files to that folder. They can be downloaded through the link below.

Once preprocessed data sets are ready to go, run the notebook (GPU runtime is recommanded since it's way faster).

## How to run experiments on local machine?
Links for downloading data,

Music audios: https://www.dropbox.com/s/wbe8jzveqw63143/genres.zip?dl=0

Preprocessed data: https://www.dropbox.com/s/tuzyvlunjiys9db/saved_dataset.zip?dl=0

Download and unzip them.

Install packages, `pip install -r requirements.txt `

To generate Mel-Spectrogram images from audios, and split training and testing data for experiments,
run `python preprocess.py`

Make sure downloaded `genres` folder (has music audios) is in the same directory as `preprocess.py`.
(I have generated and saved these preprocessed data, you can downloaded them from the link above.)


To run experiments,
run `python MusicGenreClassification.py`

Make sure `utils.py`, `MusicGenreClassification.py`, `MusicRecNet.py` and `saved_dataset` folder (either generated or downloaded) are in the save folder.
