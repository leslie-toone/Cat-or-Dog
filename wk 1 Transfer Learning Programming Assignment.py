# # Programming assignment: Customising your Models TensorFlow 2
# ## Transfer learning

# ### Instructions

# Create a neural network model to classify images of cats and dogs, using transfer
# learning: use part of a pre-trained image classifier model (trained on ImageNet)
# as a feature extractor,and train additional new layers to perform the cats and
# dogs classification task.
# 
#### PACKAGE IMPORTS ####

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model, load_model

# #### The Dogs vs Cats dataset
#
# Use the [Dogs vs Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data),
# which was used for a 2013 Kaggle competition. It consists of 25000 images containing either a cat or a dog. We will
# only use a subset of 600 images and labels. The dataset is a subset of a much larger dataset of 3 million photos
# that were originally used as a CAPTCHA (Completely Automated Public Turing test to tell Computers and Humans
# Apart), referred to as “Asirra” or Animal Species Image Recognition for Restricting Access.
#
# * J. Elson, J. Douceur, J. Howell, and J. Saul. "Asirra: A CAPTCHA that Exploits Interest-Aligned Manual Image
# Categorization." Proceedings of 14th ACM Conference on Computer and Communications Security (CCS), October 2007.
#
# Goal: train a classifier model using part of a pre-trained image classifier, using the principle of
# transfer learning.

# #### Load and preprocess the data

images_train = np.load('Customising your Models TensorFlow 2/Data/images_train.npy') / 255.
images_valid = np.load('Customising your Models TensorFlow 2/Data/images_valid.npy') / 255.
images_test = np.load('Customising your Models TensorFlow 2/Data/images_test.npy') / 255.

labels_train = np.load('Customising your Models TensorFlow 2/Data/labels_train.npy')
labels_valid = np.load('Customising your Models TensorFlow 2/Data/labels_valid.npy')
labels_test = np.load('Customising your Models TensorFlow 2/Data/labels_test.npy')

print("{} training data examples".format(images_train.shape[0]))
print("{} validation data examples".format(images_valid.shape[0]))
print("{} test data examples".format(images_test.shape[0]))

print(images_train.shape, images_valid.shape, images_test.shape)
# #### Display sample images and labels from the training set

# Display a few images and labels

class_names = np.array(['Dog', 'Cat'])

plt.figure(figsize=(15, 10))
inx = np.random.choice(images_train.shape[0], 15, replace=False)
for n, i in enumerate(inx):
    ax = plt.subplot(3, 5,
                     n + 1)  # nrows,ncols, index (plot number starts at 1 increments across rows first and has a max
    # of nrow*ncols)
    plt.imshow(images_train[i])
    plt.title(class_names[labels_train[i]])
    plt.axis('off')


# #### Create a benchmark model
#
# We will first train a CNN classifier model as a benchmark model before implementing the transfer learning approach.
# Using the functional API, build the benchmark model according to the following specifications:
#
# * The model should use the `input_shape` in the function argument to set the shape in the Input layer.
# * The first and second hidden layers should be Conv2D layers with 32 filters, 3x3 kernel size and ReLU activation.
# * The third hidden layer should be a MaxPooling2D layer with a 2x2 window size.
# * The fourth and fifth hidden layers should be Conv2D layers with 64 filters, 3x3 kernel size and ReLU activation.
# * The sixth hidden layer should be a MaxPooling2D layer with a 2x2 window size.
# * The seventh and eighth hidden layers should be Conv2D layers with 128 filters, 3x3 kernel size and ReLU activation.
# * The ninth hidden layer should be a MaxPooling2D layer with a 2x2 window size.
# * This should be followed by a Flatten layer, and a Dense layer with 128 units and ReLU activation
# * The final layer should be a Dense layer with a single neuron and sigmoid activation.
# * All of the Conv2D layers should use `'SAME'` padding.
#
# In total, the network should have 13 layers (including the `Input` layer).
#
# The model should then be compiled with the RMSProp optimiser with learning rate 0.001, binary cross entropy loss
# and and binary accuracy metric.

def get_benchmark_model(input_shape):
    inputs = Input(shape=input_shape, name='input_layer')
    h = Conv2D(32, (3, 3), activation='relu', name='Conv2D_layer1', padding='same')(inputs)
    h = Conv2D(32, (3, 3), activation='relu', name='Conv2D_layer2', padding='same')(h)
    h = MaxPooling2D((2, 2), name='maxpool2d_layer3')(h)
    h = Conv2D(64, (3, 3), activation='relu', name='Conv2D_layer4', padding='same')(h)
    h = Conv2D(64, (3, 3), activation='relu', name='Conv2D_layer5', padding='same')(h)
    h = MaxPooling2D((2, 2), name='maxpool2d_layer6')(h)
    h = Conv2D(128, (3, 3), activation='relu', name='Conv2D_layer7', padding='same')(h)
    h = Conv2D(128, (3, 3), activation='relu', name='Conv2D_layer8', padding='same')(h)
    h = MaxPooling2D((2, 2), name='maxpool2d_layer9')(h)
    h = Flatten(name='flatten_layer10')(h)
    h = Dense(128, activation='relu', name='Dense_layer11')(h)
    outputs = Dense(1, activation='sigmoid', name='Out_Dense_sigmoid_layer12')(h)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
                  loss='binary_crossentropy',
                  metrics=["acc"])
    '''According to tf.keras.Model.compile() documentation: When you pass the strings 'accuracy' or 'acc', 
    we convert this to one of tf.keras.metrics.BinaryAccuracy, tf.keras.metrics.CategoricalAccuracy, 
    tf.keras.metrics.SparseCategoricalAccuracy based on the loss function used and the model output shape. 
    We do a similar conversion for the strings 'crossentropy' and 'ce' as well.
    
    thus we can keep metric=acc rather than binary accuracy'''

    return model


# Build and compile the benchmark model, and display the model summary

benchmark_model = get_benchmark_model(images_train[0].shape)
# use the binary crossentropy loss since this is a binary classification problem

benchmark_model.summary()

# #### Train the CNN benchmark model
# 
# We will train the benchmark CNN model using an `EarlyStopping` callback.

# Fit the benchmark model and save its training history

earlystopping = tf.keras.callbacks.EarlyStopping(patience=2)

history_benchmark = benchmark_model.fit(images_train, labels_train, epochs=10, batch_size=32,
                                        validation_data=(images_valid, labels_valid),
                                        callbacks=[earlystopping])

# #### Plot the learning curves

# Run this cell to plot accuracy vs epoch and loss vs epoch

plt.figure(figsize=(15, 5))
plt.subplot(121)
try:
    plt.plot(history_benchmark.history['accuracy'])
    plt.plot(history_benchmark.history['val_accuracy'])
except KeyError:
    plt.plot(history_benchmark.history['acc'])
    plt.plot(history_benchmark.history['val_acc'])
plt.title('Accuracy vs. epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')

plt.subplot(122)
plt.plot(history_benchmark.history['loss'])
plt.plot(history_benchmark.history['val_loss'])
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

# #### Evaluate the benchmark model
# Evaluate the benchmark model on the test set

benchmark_test_loss, benchmark_test_acc = benchmark_model.evaluate(images_test, labels_test, verbose=0)
print("Test loss: {}".format(benchmark_test_loss))
print("Test accuracy: {}".format(benchmark_test_acc))


# #### Load the pretrained image classifier model
# 
#We will now begin to build our image classifier using transfer learning. Use the pre-trained MobileNet V2
# model, available to download from [Keras Applications](https://keras.io/applications/#mobilenetv2).

def load_pretrained_MobileNetV2(path):
    model = load_model(path)
    return model


# Call the function loading the pretrained model and display its summary

base_model = load_pretrained_MobileNetV2('Customising your Models TensorFlow 2/models/MobileNetV2.h5')
base_model.summary()

# #### Use the pre-trained model as a feature extractor
# 
# Remove the final layer of the network and replace it with new, untrained classifier layers for our task.
# First create a new model that has the same input tensor as the MobileNetV2 model, and use the output
# tensor from the layer with name `global_average_pooling2d_6` as the model output.

def remove_head(pretrained_model):
    model = Model(inputs=pretrained_model.input,
                  outputs=pretrained_model.get_layer('global_average_pooling2d_6').output)
    return model

feature_extractor = remove_head(base_model)
feature_extractor.summary()


# Construct new final classifier layers for your model. Using the Sequential API, create a new model
# according to the following specifications:
# 
# * The new model should begin with the feature extractor model.
# * This should then be followed with a new dense layer with 32 units and ReLU activation function.
# * This should be followed by a dropout layer with a rate of 0.5.
# * Finally, this should be followed by a Dense layer with a single neuron and a sigmoid activation function.
# 
# In total, the network should be composed of the pretrained base model plus 3 layers.


def add_new_classifier_head(feature_extractor_model):

    model = Sequential([
        feature_extractor_model,
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model


# Call the function adding a new classification head and display the summary

new_model = add_new_classifier_head(feature_extractor)
new_model.summary()


# #### Freeze the weights of the pretrained model

# Freeze the weights of the pre-trained feature extractor (defined as the first layer of the model),
# so that only the weights of the new layers added will change during the training.
# 
# Compile model as before: use the RMSProp optimiser with learning rate 0.001, binary cross
# entropy loss and and binary accuracy metric.

def freeze_pretrained_weights(model):

    model.layers[0].trainable = False
    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
                  loss='binary_crossentropy',
                  metrics=["acc"])
    return model


# Call the function freezing the pretrained weights and display the summary
print(new_model.layers)
frozen_new_model = freeze_pretrained_weights(new_model)
frozen_new_model.summary()

n_trainable_variables = len(frozen_new_model.trainable_variables)
n_non_trainable_variables = len(frozen_new_model.non_trainable_variables)

print("\n After freezing:\n\t Number of trainable variables: ", len(frozen_new_model.trainable_variables),
      "\n\t Number of non trainable variables: ", len(frozen_new_model.non_trainable_variables))

# #### Train the model
# 
# Train the new model on the dogs vs cats data subset. We will use an `EarlyStopping` callback
# with patience set to 2 epochs, as before.


# Train the model and save its training history

earlystopping = tf.keras.callbacks.EarlyStopping(patience=2)
history_frozen_new_model = frozen_new_model.fit(images_train, labels_train, epochs=10, batch_size=32,
                                                validation_data=(images_valid, labels_valid),
                                                callbacks=[earlystopping])

# #### Plot the learning curves

# Run this cell to plot accuracy vs epoch and loss vs epoch

plt.figure(figsize=(15, 5))
plt.subplot(121)
try:
    plt.plot(history_frozen_new_model.history['accuracy'])
    plt.plot(history_frozen_new_model.history['val_accuracy'])
except KeyError:
    plt.plot(history_frozen_new_model.history['acc'])
    plt.plot(history_frozen_new_model.history['val_acc'])
plt.title('Accuracy vs. epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')

plt.subplot(122)
plt.plot(history_frozen_new_model.history['loss'])
plt.plot(history_frozen_new_model.history['val_loss'])
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

# #### Evaluate the new model

# Evaluate the benchmark model on the test set

new_model_test_loss, new_model_test_acc = frozen_new_model.evaluate(images_test, labels_test, verbose=0)
print("Test loss: {}".format(new_model_test_loss))
print("Test accuracy: {}".format(new_model_test_acc))

# #### Compare both models
# 
# Finally, we will look at the comparison of training, validation and test metrics between the benchmark and transfer
# learning model.

# Gather the benchmark and new model metrics

benchmark_train_loss = history_benchmark.history['loss'][-1]
benchmark_valid_loss = history_benchmark.history['val_loss'][-1]

try:
    benchmark_train_acc = history_benchmark.history['acc'][-1]
    benchmark_valid_acc = history_benchmark.history['val_acc'][-1]
except KeyError:
    benchmark_train_acc = history_benchmark.history['accuracy'][-1]
    benchmark_valid_acc = history_benchmark.history['val_accuracy'][-1]

new_model_train_loss = history_frozen_new_model.history['loss'][-1]
new_model_valid_loss = history_frozen_new_model.history['val_loss'][-1]

try:
    new_model_train_acc = history_frozen_new_model.history['acc'][-1]
    new_model_valid_acc = history_frozen_new_model.history['val_acc'][-1]
except KeyError:
    new_model_train_acc = history_frozen_new_model.history['accuracy'][-1]
    new_model_valid_acc = history_frozen_new_model.history['val_accuracy'][-1]

# Compile the metrics into a pandas DataFrame and display the table

comparison_table = pd.DataFrame([['Training loss', benchmark_train_loss, new_model_train_loss],
                                 ['Training accuracy', benchmark_train_acc, new_model_train_acc],
                                 ['Validation loss', benchmark_valid_loss, new_model_valid_loss],
                                 ['Validation accuracy', benchmark_valid_acc, new_model_valid_acc],
                                 ['Test loss', benchmark_test_loss, new_model_test_loss],
                                 ['Test accuracy', benchmark_test_acc, new_model_test_acc]],
                                columns=['Metric', 'Benchmark CNN', 'Transfer learning CNN'])
comparison_table.index = [''] * 6
print(comparison_table)

# Plot confusion matrices for benchmark and transfer learning models

plt.figure(figsize=(15, 5))

preds = benchmark_model.predict(images_test)
preds = (preds >= 0.5).astype(np.int32)
cm = confusion_matrix(labels_test, preds)
df_cm = pd.DataFrame(cm, index=['Dog', 'Cat'], columns=['Dog', 'Cat'])
plt.subplot(121)
plt.title("Confusion matrix for benchmark model\n")
sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
plt.ylabel("Predicted")
plt.xlabel("Actual")

preds = frozen_new_model.predict(images_test)
preds = (preds >= 0.5).astype(np.int32)
cm = confusion_matrix(labels_test, preds)
df_cm = pd.DataFrame(cm, index=['Dog', 'Cat'], columns=['Dog', 'Cat'])
plt.subplot(122)
plt.title("Confusion matrix for transfer learning model\n")
sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
plt.ylabel("Predicted")
plt.xlabel("Actual")
plt.show()
