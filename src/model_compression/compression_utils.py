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

def convert_to_tflite_compress(filename):

  model = keras.models.load_model('models/active/' + filename)

  # Convert the model to the TensorFlow Lite format without quantization
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
  ]

  model_tflite = converter.convert()

  output_file = "models/active/" + filename + "_compressed" + ".tflite"

  open(output_file, "wb").write(model_tflite)

  return output_file


def convert_to_tflite_quanitzed(filename):

  model = keras.models.load_model('models/active/' + filename)

  # Convert the model to the TensorFlow Lite format with quantization
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
  ]

  converter.optimizations = [tf.lite.Optimize.DEFAULT] # Quantize

  model_tflite = converter.convert()

  output_file = "models/active/" + filename + "_quantized" + ".tflite"

  open(output_file, "wb").write(model_tflite)

  return output_file

def get_confusion_matrix(filename, y_real, y_preds):

  class_names = np.hstack((np.arange(1,13), np.arange(41,50)))

  cm = confusion_matrix(y_real, np.asarray(y_preds))

  fig = plt.figure(figsize=(16, 14))
  ax= plt.subplot()
  sns.heatmap(cm, annot=True, ax = ax, fmt = 'g'); #annot = True to annotate cells

  ax.set_xlabel('Predicted', fontsize=20)
  ax.xaxis.set_label_position('bottom')
  plt.xticks(rotation=90)
  ax.xaxis.set_ticklabels(class_names, fontsize = 10)
  ax.xaxis.tick_bottom()

  ax.set_ylabel('True', fontsize=20)
  ax.yaxis.set_ticklabels(class_names, fontsize = 10)
  plt.yticks(rotation=0)

  plt.title('Ninapro DB2: Confusion Matrix', fontsize=20)

  plt.savefig('images/active/confusion_matrix_' + filename.split('/')[-1].split('.')[0] +'.png')


def get_model_accuracy(model_path, test_x, test_y, conf=True):

  # Get interpreter
  interpreter = tf.lite.Interpreter(model_path=model_path)
  interpreter.allocate_tensors()

  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  input_shape = input_details[0]['shape']

  # Get test data
  test_x = np.load(test_x)
  test_y = np.load(test_y)

  # Restrict to desired gestures
  inds = np.argwhere(np.logical_or(np.argmax(test_y, axis=1) < 13, np.argmax(test_y, axis=1) > 40)).squeeze()
  test_x = test_x[inds, :, :]
  test_y = np.hstack((test_y[inds, :13], test_y[inds, 41:]))

  y_real = np.argmax(test_y, axis=1)

  count_correct = 0

  y_preds = []

  for i in range(test_x.shape[0]):
    
    start = time.time()

    # Use same image as Keras model
    input_data = np.expand_dims(test_x[i], axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    end = time.time()

    print("Inference time:", end - start)

    pred = np.argmax(output_data)
    real = y_real[i]

    y_preds.append(pred)

    if pred == real:
      count_correct += 1

  if conf:
    get_confusion_matrix(model_path, y_real, y_preds)

  return (count_correct/test_x.shape[0])*100