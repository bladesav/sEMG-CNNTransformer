import os
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
import seaborn as sns
import time
import argparse

from compression_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compressing model.")

    filename_default = '_'.join([str(1), str(True), str(1000), str(1) + 'none' + str(1), str(False), str(False)])
    parser.add_argument("filename", type=str, help="File name of model to compress/quantize.")
    parser.add_argument("-x", type=str, help="File name of test x data.", default="data/processed/" + filename_default + '_test_x' + ".npy")
    parser.add_argument("-y", type=str, help="File name of test y data.", default="data/processed/" + filename_default + '_test_y' + ".npy")
    parser.add_argument("-e", "--evaluate", type=bool, help="Evaluate compressed/quantized model and generate confusion matrix.", default=True)
    parser.add_argument("-c", "--compress", type=bool, help="Generate compressed model.", default=True)
    parser.add_argument("-q", "--quantize", type=bool, help="Generate quantized model.", default=True)
    args = parser.parse_args()

    if args.compress:
        out = convert_to_tflite_compress(args.filename)
        if args.evaluate:
            get_model_accuracy(out, args.x, args.y)

    if args.quantize:
        out = convert_to_tflite_quanitzed(args.filename)
        if args.evaluate:
            get_model_accuracy(out, args.x, args.y)