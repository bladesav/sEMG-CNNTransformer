import os
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
import seaborn as sns

from training_utils import *

# Set seed for experiment reproducibility
seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing code.")

    # Mandatory metadata
    parser.add_argument("subject", type=int, help="Subject desired for training.")
    parser.add_argument("runid", type=int, help="Slurm run ID.")

    # Data arguments
    filename_default = '_'.join([str(1), str(True), str(1000), str(1) + 'none' + str(1), str(False), str(False)])
    parser.add_argument("-train_x", type=str, help="Manual filename of x train data.", default=filename_default + '_train_x' + ".npy")
    parser.add_argument("-train_y", type=str, help="Manual filename  of y train data.", default=filename_default + '_train_y' + ".npy")
    parser.add_argument("-test_x", type=str, help="Manual filename  of x test data.", default=filename_default + '_test_x' + ".npy")
    parser.add_argument("-test_y", type=str, help="Manual filename  of y test data.", default=filename_default + '_test_y' + ".npy")

    # Model arguments deep_layers=2, stride_increase=0, expansion=1, depthwise=1, outputchannels=1)
    parser.add_argument("-dl", "--deep_layers", type=int, default=2)
    parser.add_argument("-s", "--stride_increase", type=int, default=0)
    parser.add_argument("-e", "--expansion", type=int, default=1)
    parser.add_argument("-dw", "--depthwise", type=int, default=1)
    parser.add_argument("-oc", "--outputchannels", type=int, default=1)
    
    # Training arguments 
    parser.add_argument("-m", "--model_type", type=str, default="CNNAttention")
    parser.add_argument("-desc", "--description", type=str, default="Vanilla")
    parser.add_argument("-ep", "--epochs", type=int, default=10)
    parser.add_argument("-bs", "--batch_size", type=int, default=32)

    args = parser.parse_args()

    # Load data and select desired gestures
    train_x = np.load("data/processed/" + args.train_x)
    train_y = np.load("data/processed/" + args.train_y)

    inds = np.argwhere(np.logical_or(np.argmax(train_y, axis=1) < 13, np.argmax(train_y, axis=1) > 40)).squeeze()
    train_x = train_x[inds, :, :]
    train_y = np.hstack((train_y[inds, :13], train_y[inds, 41:]))

    test_x = np.load("data/processed/" + args.test_x)
    test_y = np.load("data/processed/" + args.test_y)

    inds = np.argwhere(np.logical_or(np.argmax(test_y, axis=1) < 13, np.argmax(test_y, axis=1) > 40)).squeeze()
    test_x = test_x[inds, :, :]
    test_y = np.hstack((test_y[inds, :13], test_y[inds, 41:]))

    # Get model and train
    model = get_CNNAttention(deep_layers=args.deep_layers, stride_increase=args.stride_increase, expansion=args.expansion, 
    depthwise=args.depthwise, outputchannels=args.outputchannels)

    train(model, args.runid, str(args.subject), args.model_type, args.description, args.epochs, args.batch_size, train_x, train_y, test_x, test_y)
    