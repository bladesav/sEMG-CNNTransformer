import os
import zipfile
from scipy.io import loadmat
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse

from preprocess_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing code.")
    parser.add_argument("-n", "--norm", type=bool, help="Either True (z-score) or False (none).", default=True)
    parser.add_argument("-fq", "--freq", type=int, help="Desired frequency.", default=1000)
    parser.add_argument("-ft", "--filt", type=str, help="Either 'lowpass', 'highpass', 'none'.", default='none')
    parser.add_argument("-rc", "--rmscont", type=bool, help="Continuous RMS thresholding; either True or False.", default=False)
    parser.add_argument("-rnc", "--rmsnoncont", type=bool, help="Non-continuous RMS thresholding; either True or False.", default=False)
    parser.add_argument("-u", "--unzip", type=bool, help="Unzips zipped Ninapro files.", default=False)
    parser.add_argument("-fc", "--filtcut", type=int, help="Cutoff frequency for filter.", default=1)
    parser.add_argument("-fo", "--filtord", type=int, help="Filter order.", default=1)
    args = parser.parse_args()
    
    if args.unzip:
        unzip_data()

    # for subject in range(40):
    #     try:
    get_exercise_data(1, args.norm, args.freq, args.filt, args.rmscont, args.rmsnoncont, filtcut=args.filtcut, filtord=args.filtord)
        # except:
        #     print(f"ERROR: Could not process Subject {subject} data.")
        #     continue