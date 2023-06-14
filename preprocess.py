import numpy as np
import pandas as pd
import os
from scipy.io import wavfile
import scipy.signal as signal
import numpy as np
import wandb
import librosa


def read_data(path_to_data, hyperparams, wandb=False):
    # List all the folders in the path_to_data directory
    folders = os.listdir(path_to_data)

    # Initialize lists to store file paths and corresponding labels
    file_paths = []
    labels = []

    # Loop through the folders
    for folder in folders:
        # Get the full folder path
        folder_path = os.path.join(path_to_data, folder)

        # Check if the folder is a directory
        if os.path.isdir(folder_path):
            # Get the label from the folder name
            label = folder

            # Get the file names in the folder
            files = os.listdir(folder_path)

            # Loop through the files
            for file in files:
                # Get the file path
                file_path = os.path.join(folder_path, file)

                # Append the file path and label to the lists
                file_paths.append(file_path)
                labels.append(label)

    # Create a dataframe with all the data
    data = pd.DataFrame({"file_path": file_paths, "label": labels})

    # Read the .wav files and store the signals and sampling rates
    data["sr"], data["signal"] = zip(*data["file_path"].map(wavfile.read))

    # Plot in wandb a random signal
    random_signal = np.random.randint(0, len(data))
    if wandb:
        wandb.log(
            {
                "signal": wandb.Audio(
                    data["signal"][random_signal],
                    sample_rate=data["sr"][random_signal],
                    caption="Random raw signal " + data["label"][random_signal],
                )
            }
        )

    # Generate ten folds for cross-validation. Store them as a label in the dataframe
    data["fold"] = data.groupby("label").cumcount() % 10

    # Check the len of each signal
    data["len"] = data["signal"].apply(len)
    data["max"] = data["signal"].apply(np.abs).apply(np.max)

    # ============== Preprocessing ===========
    # Normalize the signals applying the normalize_audio function with receives as input the signal
    data["norm_signal"] = data.apply(lambda x: normalize_audio(x["signal"]), axis=1)

    # Plot in wandb a random signal
    if wandb:
        wandb.log(
            {
                "norm_signal": wandb.Audio(
                    data["norm_signal"][random_signal],
                    sample_rate=data["sr"][random_signal],
                    caption="Random normalised signal" + data["label"][random_signal],
                )
            }
        )

    # Frame the signals applying the frame_audio function with receives as input the signal, the frame size, the hop size and the sampling rate
    data["framed_signal"] = data.apply(
        lambda x: frame_audio(
            x["norm_signal"],
            hyperparams["frame_size_ms"],
            hyperparams["hop_size_percent"],
            x["sr"],
        ),
        axis=1,
    )

    # Plot in wandb a random chunk of a signal
    if wandb:
        random_frame = np.random.randint(0, len(data["framed_signal"][random_signal]))
        wandb.log(
            {
                "framed_signal": wandb.Audio(
                    data["framed_signal"][random_signal][random_frame],
                    sample_rate=data["sr"][random_signal],
                    caption="Random framed signal" + data["label"][random_signal],
                )
            }
        )

    # Each row of the dataframe contains a signal. Each signal is framed into several frames. Each frame is a row in the dataframe that has to contain the label of the signal.
    # We have to duplicate the rows to have one row per frame. We can use the pandas function explode to do this.
    data_framed = data.explode("framed_signal")

    return data, data_framed


# Normalize the signals
def normalize_audio(audio_data, max=None):
    if max is None:
        max_value = np.max(np.abs(audio_data))
    else:
        max_value = max
    normalized_data = audio_data / max_value
    return normalized_data


def frame_audio(audio_data, frame_size_ms, hop_size_percent, sample_rate):
    frame_size = int(frame_size_ms * sample_rate / 1000)
    hop_size = int(frame_size * hop_size_percent / 100)
    num_samples = len(audio_data)
    num_frames = int(np.ceil((num_samples - frame_size) / hop_size) + 1)

    framed_data = np.zeros((num_frames, frame_size))

    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        frame = audio_data[start:end]
        if len(frame) < frame_size:
            frame = np.append(frame, np.zeros(frame_size - len(frame)))
        else:  # TODO: preguntar si descartamos la que se queda corta o la relleno con 0s
            frame = np.hanning(frame_size) * frame
        framed_data[i] = frame

    return framed_data


def antisymmetric_fir_filter(f):
    coefficients = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9])
    # Reverse the coefficients and negate the odd-indexed elements
    antisymmetric_coeffs = np.flipud(coefficients)
    antisymmetric_coeffs[1::2] *= -1

    # Apply the FIR filter
    filtered_signal = signal.convolve(f, antisymmetric_coeffs, mode="same")

    return filtered_signal


def cms(mfcc):
    # Compute the mean of each feature
    mean = np.mean(mfcc, axis=0)

    # Apply CMN
    cmn_mfcc = mfcc - mean

    return cmn_mfcc


def extract_mfcc_with_derivatives(frame, sample_rate):
    # Antisimetric filter
    frame = antisymmetric_fir_filter(frame)
    # Compute MFCC coefficients
    mfcc_coeffs = librosa.feature.mfcc(y=frame, sr=sample_rate, n_mfcc=10)
    # Compute cepstral mean substraction
    mfcc_coeffs = cms(mfcc_coeffs)

    # Compute delta coefficients
    mfcc_delta = librosa.feature.delta(mfcc_coeffs)
    # Compute delta-delta coefficients
    mfcc_delta2 = librosa.feature.delta(mfcc_coeffs, order=2)

    # Concatenate the three coefficients
    mfcc_features = np.concatenate((mfcc_coeffs, mfcc_delta, mfcc_delta2), axis=0)

    return mfcc_features
