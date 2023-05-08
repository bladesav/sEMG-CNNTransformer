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

#################################
# AUXILARY FUNCTIONS FOR MODELS #
#################################

def evaluate_metrics(model, model_name, test_x, test_y):
  
  x = test_x
  y = test_y

  loss, acc = model.evaluate(x, y, verbose=2)

  y_preds = np.argmax(model.predict(x), axis=1)

  f1 = f1_score(np.argmax(y, axis=1), y_preds, average=None)

  print(loss, acc, f1)

  class_names = np.hstack((np.arange(1,13), np.arange(41,50)))

  cm = confusion_matrix(np.argmax(y, axis=1), y_preds)

  fig = plt.figure(figsize=(16, 14))
  ax= plt.subplot()
  sns.heatmap(cm, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells

  ax.set_xlabel('Predicted', fontsize=20)
  ax.xaxis.set_label_position('bottom')
  plt.xticks(rotation=90)
  ax.xaxis.set_ticklabels(class_names, fontsize = 10)
  ax.xaxis.tick_bottom()

  ax.set_ylabel('True', fontsize=20)
  ax.yaxis.set_ticklabels(class_names, fontsize = 10)
  plt.yticks(rotation=0)

  plt.title('Ninapro DB2: Confusion Matrix', fontsize=20)

  plt.savefig('images/active/confusion_matrix_'+ model_name + "_" + str(round(acc, 4)) +'.png')

  return acc

def conv_block(x, filters=1, kernel_size=3, strides=[10, 1]):

    x = layers.Conv2D(
        filters, kernel_size, strides=strides, activation=tf.nn.swish, padding="same"
    )(x)
    x = layers.Dropout(0.3)(x)

    return x

def inverted_residual_block(x, expanded_channels, output_channels, strides=1):
    # m = layers.Conv2D(expanded_channels, 1, padding="same", use_bias=False)(x)
    # m = layers.BatchNormalization()(m)
    # m = tf.nn.swish(m)

    # if strides == 2:
    #     m = layers.ZeroPadding2D(padding=imagenet_utils.correct_pad(m, 3))(m)
    m = layers.DepthwiseConv2D(
        12, strides=strides, padding="same" if strides == 1 else "valid", use_bias=False
    )(m)
    # m = layers.BatchNormalization()(m)
    # m = tf.nn.swish(m)

    # m = layers.Conv2D(output_channels, 1, padding="same", use_bias=False)(x)
    # m = layers.BatchNormalization()(m)
    # m = tf.nn.swish(m)

    if tf.math.equal(x.shape[-1], output_channels) and strides == 1:
        return layers.Add()([m, x])
    return m

def cnnatt_conv_block(x, filters=16, kernel_size=10, strides=10):

    x = layers.Conv1D(
        filters, kernel_size, strides=strides, activation=tf.nn.swish, padding="same"
    )(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(0.3)(x)

    return x

# Reference: https://git.io/JKgtC
def cnnatt_inverted_residual_block(x, expanded_channels, output_channels, strides=1, depth_filts=6):
    m = layers.Conv1D(expanded_channels, 1, padding="same", use_bias=False)(x)
    m = layers.LayerNormalization(epsilon=1e-6)(m)
    m = tf.nn.swish(m)

    # if strides == 2:
    #     m = layers.ZeroPadding2D(padding=imagenet_utils.correct_pad(m, 3))(m)
    m = layers.DepthwiseConv1D(
        depth_filts, strides=strides, padding="same" if strides == 1 else "valid", use_bias=False
    )(m)
    m = layers.LayerNormalization(epsilon=1e-6)(m)
    m = tf.nn.swish(m)

    m = layers.Conv1D(output_channels, 1, padding="same", use_bias=False)(m)
    m = layers.LayerNormalization(epsilon=1e-6)(m)

    # m = mlp(m, hidden_units=[m.shape[-1] * 2, m.shape[-1]], dropout_rate=0.4)

    if tf.math.equal(x.shape[-1], output_channels) and strides == 1:
        return layers.Add()([m, x])
    return m


# Reference:
# https://keras.io/examples/vision/image_classification_with_vision_transformer/
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.swish)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def transformer_block(x, transformer_layers, projection_dim, num_heads=2):
    for _ in range(transformer_layers):

        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.4
        )(x1, x1)

        x2 = layers.Add()([attention_output, x])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=[x.shape[-1] * 2, x.shape[-1]], dropout_rate=0.4)
        x = layers.Add()([x3, x2])

    return x


def mobilevit_block(x, num_blocks, projection_dim, patch_size=3, strides=1):
    # Local projection with convolutions.
    local_features = conv_block(x, filters=projection_dim, strides=strides)
    local_features = conv_block(
        local_features, filters=projection_dim, kernel_size=1, strides=strides
    )

    # Unfold into patches and then pass through Transformers.
    num_patches = int((local_features.shape[1] * local_features.shape[2]) / patch_size)

    non_overlapping_patches = layers.Reshape((patch_size, num_patches, projection_dim))(
        local_features
    )

    global_features = transformer_block(
        non_overlapping_patches, num_blocks, projection_dim
    )

    # Fold into conv-like feature-maps.
    folded_feature_map = layers.Reshape((*local_features.shape[1:-1], projection_dim))(
        global_features
    )

    # Apply point-wise conv -> concatenate with the input features.
    folded_feature_map = conv_block(
        folded_feature_map, filters=x.shape[-1], kernel_size=1, strides=strides
    )
    local_global_features = layers.Concatenate(axis=-1)([x, folded_feature_map])

    # Fuse the local and global features using a convoluion layer.
    local_global_features = conv_block(
        local_global_features, filters=projection_dim, strides=strides
    )

    return local_global_features


###############
# MODEL TYPES #
###############

def get_MViT(patch_size=3, blocks=1, deep_layers=1, expansion_factor=2, width=12, depth=150):

    inputs = keras.Input((12, 12, 12))

    # Initial conv-stem -> MV2 block.
    x = conv_block(inputs, filters=16)

    # First MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=16 * expansion_factor, output_channels=24, strides=1
    )
    x = mobilevit_block(x, num_blocks=2, patch_size=3, projection_dim=48)

    # Second MV2 -> MobileViT block.
    if blocks > 1:
        x = inverted_residual_block(
            x, expanded_channels=48 * expansion_factor, output_channels=48, strides=1
        )
        x = mobilevit_block(x, num_blocks=2, projection_dim=64, patch_size=3)

    # Third MV2 -> MobileViT block.
    if blocks > 2:
        x = inverted_residual_block(
            x, expanded_channels=64 * expansion_factor, output_channels=64, strides=1
        )
        x = mobilevit_block(x, num_blocks=1, projection_dim=80, patch_size=3)

        x = inverted_residual_block(
            x, expanded_channels=80 * expansion_factor, output_channels=80, strides=1
        )
        x = mobilevit_block(x, num_blocks=1, projection_dim=96, patch_size=3)

    # Classification head.
    x = layers.GlobalAvgPool2D()(x)

    if deep_layers > 3:
        x = layers.Dense(2048)(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(1024)(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(512)(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

    if deep_layers > 2:
        x = layers.Dense(256)(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

    if deep_layers > 1:
        x = layers.Dense(128)(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

    if deep_layers > 0:
        x = layers.Dense(64)(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(50, activation="softmax")(x)

    return keras.Model(inputs, outputs)

def get_CNNAttention(width=12, depth=150, deep_layers=2, stride_increase=0, 
expansion=1, depthwise=1, outputchannels=1):

    inputs = keras.Input((depth, width))

    # Stage 1
    x = cnnatt_conv_block(inputs, kernel_size=4, strides=(3 + stride_increase), filters=(outputchannels*24))
    x = cnnatt_inverted_residual_block(x, expanded_channels=(3 + expansion)*(outputchannels*24), output_channels=(outputchannels*24), depth_filts=(5+depthwise))
    x = transformer_block(x, transformer_layers=1, num_heads=1, projection_dim=(outputchannels*24))
    x = layers.Dropout(0.5)(x)

    # Stage 2
    x = cnnatt_conv_block(x, kernel_size=6, strides=(4 + stride_increase), filters=(outputchannels*48))
    x = cnnatt_inverted_residual_block(x, expanded_channels=(5 + expansion)*(outputchannels*48), output_channels=(outputchannels*48), depth_filts=(8+depthwise))
    x = transformer_block(x, transformer_layers=1, num_heads=3, projection_dim=(outputchannels*48))
    x = layers.Dropout(0.5)(x)

    # Classification head.
    x = layers.GlobalAvgPool1D()(x)

    if deep_layers > 1:
        x = layers.Dense(128)(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

    if deep_layers > 0:
        x = layers.Dense(32)(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(22, activation="softmax")(x)

    return keras.Model(inputs, outputs)

#####################
# TRAINING FUNCTION #
#####################

def train(model, runid, subject, model_type, description, epochs, batch_size, train_x, train_y, test_x, test_y):

    # Model details
    model.summary()

    # Naming convention is {RUNID}_{SUBJECT}_{MODEL}_{EXTRADESC}_{ACCURACY}.h5
    model_name = '_'.join([str(runid), str(subject), model_type, description])

    # Compile model for training and specify hyperparameters
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])

    checkpoint_path = "models/checkpoints/" + model_name

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    verbose=1)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=5, min_lr=0.0001)

    # Fit data to model

    print(train_x.shape, train_y.shape)
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
                    validation_data=(test_x, test_y), callbacks=[cp_callback, reduce_lr])

    # Evaluate and save model
    acc = evaluate_metrics(model, model_name, test_x, test_y)
    model_save_path = "models/active/" + model_name + "_" + str(round(acc, 4)) + ".h5"

    model.save(model_save_path)