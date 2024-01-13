import os
import math
import json
import librosa
# run in terminal: pip install librosa

DATASET_PATH = "genre_dataset"
JSON_PATH = "data.json"

SAMPLE_RATE = 22050 # num of samples per second (Hz) while digitalizing audio signal
DURATION = 30 # in seconds. this requires that all songs in dataset are of the same duration
NUM_OF_SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION # in this case 661 500

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2028, hop_length=512, num_segments=5):
    """
    num_sugments parameter нужен для того чтобы делить
    треки на сегменты,чтобы увеличить объём датасета.
    вместо нескольких длинных треков использовать много коротких
    другие параметры объяснены в конспекте 
    https://github.com/arimaz1881/ML-Exercises/blob/master/Spectrogram_Audio_data_preprocessing/Spectrogram_Audio_data_preprocessing.ipynb
    """

    # dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    num_samples_per_segment = int(NUM_OF_SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors = math.ceil(num_samples_per_segment / hop_length) # round to higher value
    """
    we are calculating mfcc after each hop_length.
    this value is necessary to ensure that number of mfcc vertors is the same
    for all track segments
    hop_length: It represents the number of samples between 
    the start of one frame and the start of the next frame. 
    """

    #loop through all the genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        """
        dirpath: the path to the folder that we are currently in
        dirnames: list of the subdirectories in the current dirpath
        filenames: list of the files stored in dirpath
        """
        # ensure that we are not at the root level
        # because we only need the names in genre folders
        if dirpath != dataset_path:

            # save semantic label into data["mapping"]
            dirpath_components = os.path.split(dirpath) # genre/blues => ["genre", "blues"]
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))

            # go through all files
            # process files for a specific genre
            for f in filenames:
                """
                file_path: to load the audio for transforming we need the full filepath
                of the audio, so os.path.join is used to concat current
                dirpath with filename for the current file
                """
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # now we need to split the signal to a bunch of segments
                for segment in range(num_segments):
                    start_sample = num_samples_per_segment * segment
                    finish_sample = start_sample + num_samples_per_segment
        
                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    # store mfcc for segment if it has the expected length
                    if len(mfcc) == expected_num_mfcc_vectors:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, segment+1))


                        """
                        data["mfcc"] accepts list of MFCCs
                        data["labels"] accepts int values from 0 to number of genres
                        we append i-1 because first iteration in os.walk we are in
                        the root directory of the dataset, which is ignored
                        """

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)



if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH)