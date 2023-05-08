import os
import zipfile
from scipy.io import loadmat
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt

def unzip_data():
  for root, dirs, files in os.walk("data/zipped", topdown=False):
    for name in files:
        if name.endswith('.zip'):
            folder_name = "data/raw/" + name.split('.')[0] + '/'
            print("Unzipping " + folder_name + " ...")
            try:
                with zipfile.ZipFile(os.path.join(root, name), 'r') as zip_ref:
                    zip_ref.extractall(folder_name)
            except:
                print("ERROR: Could not unzip " + os.path.join(root, name) + ".")


def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    from scipy.signal import butter, lfilter, filtfilt

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='bandstop')
    y = filtfilt(b, a, data)
    return y


def butter_lowpass_filter(data, cut, fs, order):
    from scipy.signal import butter, lfilter, filtfilt

    nyq = 0.5 * fs
    cut = cut / nyq

    b, a = butter(order, cut, btype='low')
    y = filtfilt(b, a, data)
    return y


def butter_highpass_filter(data, cut, fs, order):
    from scipy.signal import butter, lfilter, filtfilt

    nyq = 0.5 * fs
    cut = cut / nyq

    b, a = butter(order, cut, btype='high')
    y = filtfilt(b, a, data)
    return y


def ninapro_channel_preprocess(channel, filt, filtcut, filtord):

  if filt == 'lowpass':
    channel = butter_lowpass_filter(channel, filtcut, 2000, filtord)

  elif filt == 'highpass':
    channel = butter_highpass_filter(channel, filtcut, 2000, filtord)

  return channel


def normalize(train_data, test_data):
  train_mean = np.mean(train_data)
  train_std = np.std(train_data)
  return (train_data - train_mean) / train_std, (test_data - train_mean) / train_std


def continuous_segments(label):

  label = np.asarray(label)

  if not len(label):
      return

  breaks = list(np.where(label[:-1] != label[1:])[0] + 1)
  for begin, end in zip([0] + breaks, breaks + [len(label)]):
      assert begin < end
      yield begin, end


def find_activity_segments(data, window = 150):

  data = data[:len(data) // window * window].reshape(-1, window, 12)

  rms = np.sqrt(np.mean(np.square(data), axis=1))

  rms = np.mean(rms, axis=1)

  threshold = np.mean(rms)

  mask = rms > threshold

  for i in range(1, len(mask) - 1):
      if not mask[i] and mask[i - 1] and mask[i + 1]:
          mask[i] = True

  begin, end = max(continuous_segments(mask),
                    key=lambda s: (mask[s[0]], s[1] - s[0]))
  
  data = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2]))[begin*window : end*window, :]

  return data


def get_exercise_data(subject, norm, freq, filt, rmscont, rmsnoncont, filtcut=1, filtord=1):

  print(f"Starting subject {subject}...")
  
  for root, dirs, files in os.walk("data/raw"):

    for name in files:

        if "DB2_s" + str(subject) + "/" in root and name.endswith('.mat'):

          print(name)

          # Load data from .mat file
          annots = loadmat(os.path.join(root, name))

          for gesture in np.unique(annots['stimulus']):

            if gesture == 0:

              continue

            print(name, gesture)

            for rep in range(1, 7):

              indices = np.intersect1d(np.argwhere(annots['repetition'] == rep)[:,0], np.argwhere(annots['stimulus'] == gesture)[:,0])
              
              # Get sEMG data for subject and repetition

              emg_data = annots['emg'][indices, :][:10000, :]

              # Apply filter

              if filt != 'none':
                emg_data = np.apply_along_axis(ninapro_channel_preprocess, 0, emg_data, filt, filtcut, filtord)

              # Downsample

              inc = int(round(1/(freq/2000)))

              window_size = int(300 / inc)
              
              emg_data = emg_data[::inc]

              # Apply cont-RMS threshold 

              if rmscont:

                emg_data = find_activity_segments(emg_data, window = window_size)

              # Reshape into proper size

              emg_data = np.lib.stride_tricks.sliding_window_view(emg_data, window_size, axis = 0)

              emg_data = emg_data.reshape((emg_data.shape[0], window_size, 12))

              # Partition into train and test according to typical scheme (2nd and 5th reserved for testing)

              if rep == 2 or rep == 5:

                try:
                  test_data_x = np.vstack((test_data_x, emg_data))
                  test_data_y = np.vstack((test_data_y, np.zeros((emg_data.shape[0], 1)) + gesture))

                except:
                  test_data_x = emg_data
                  test_data_y = np.zeros((emg_data.shape[0], 1)) + gesture

              else:

                try:
                  train_data_x = np.vstack((train_data_x, emg_data))
                  train_data_y = np.vstack((train_data_y, np.zeros((emg_data.shape[0], 1)) + gesture))

                except:
                  train_data_x = emg_data
                  train_data_y = np.zeros((emg_data.shape[0], 1)) + gesture

  # Perform Z-score normalization
  
  if norm:
    train_data_x, test_data_x = normalize(train_data_x, test_data_x)

  # Apply noncont-RMS threshold 

  if rmsnoncont:

    rms_threshold = np.mean(np.sqrt(np.mean(np.square(train_data_x), axis=1)))

    train_rms_vals = np.mean(np.sqrt(np.mean(np.square(train_data_x), axis=1)), axis=1)
    test_rms_vals = np.mean(np.sqrt(np.mean(np.square(test_data_x), axis=1)), axis=1)

    train_inds = np.squeeze(np.argwhere(train_rms_vals > rms_threshold))
    test_inds = np.squeeze(np.argwhere(test_rms_vals > rms_threshold))

    train_data_x = train_data_x[train_inds, :, :]
    train_data_y = train_data_y[train_inds, :]

    test_data_x = test_data_x[test_inds, :, :]
    test_data_y = test_data_y[test_inds, :]

  # Save according to naming convention {SUBJECT}_{NORMALIZATION}_{FREQ}_{FILTER}_{RMS}_{train/test}_{x/y}.npy

  file_pref = '_'.join([str(subject), str(norm), str(freq), str(filtord) + filt + str(filtcut), str(rmscont), str(rmsnoncont)])

  file_name = file_pref + '_train_y' + ".npy"
  np.save("data/processed/" + file_name, to_categorical(train_data_y, 50))

  file_name = file_pref + '_train_x' + ".npy"
  np.save("data/processed/" + file_name, train_data_x)

  file_name = file_pref + '_test_y' + ".npy"
  np.save("data/processed/" + file_name, to_categorical(test_data_y, 50))

  file_name = file_pref + '_test_x' + ".npy"
  np.save("data/processed/" + file_name, test_data_x)