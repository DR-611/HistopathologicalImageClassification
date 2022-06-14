from google.colab import files, drive
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Rescaling, Conv2D, MaxPool2D, Flatten, Dense, Dropout


#function to build the baseline CNN model
def build_baseline_model(input_shape, K):
    model =  Sequential([
        Rescaling(1./255),
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPool2D(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D(),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(K, activation='softmax')
    ])

    return model



#function to build the dropout model (baseline model + dropout layers)
def build_dropout_model(input_shape, K):
    model = Sequential([
        Rescaling(1./255),
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        Dropout(0.2),
        MaxPool2D(),
        Conv2D(64, (3, 3), activation='relu'),
        Dropout(0.2),
        MaxPool2D(),
        Conv2D(64, (3, 3), activation='relu'),
        Dropout(0.2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(K, activation='softmax')
    ])

    return model



#function to create the data augmentation layer
def build_augmentation_layer(random_state):
    data_augmentation_layer = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(factor=0.5, seed=random_state),
        layers.RandomZoom(height_factor=(-0.3,0.3), seed=random_state),
        layers.RandomContrast(factor=0.3, seed=random_state)
    ])

    return data_augmentation_layer


#function to build the data augmentation model
def build_data_augmentation_model(input_shape, K, random_state):
    model = Sequential([
        Rescaling(1./255),
        build_augmentation_layer(random_state),
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPool2D(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D(),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(K, activation='softmax')
    ])

    return model


#function to build the deep CNN model
###Deep Model (More Convolutional Layers)

#function to build the deep CNN model
def build_deep_model(input_shape, K):
    model = Sequential([
        Rescaling(1./255),
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPool2D(),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D(),
        Conv2D(128, (3, 3), activation='relu'),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(K, activation='softmax')
    ])

    return model


#function to build the deep augmented model
def build_deep_augmented_model(input_shape, K, random_state):
    model = Sequential([
        Rescaling(1./255),
        build_augmentation_layer(random_state),
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPool2D(),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D(),
        Conv2D(128, (3, 3), activation='relu'),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(K, activation='softmax')
    ])

    return model


#function to return the ResNet50 architecture as a base model
def build_resnet50_base(input_shape):
    model =  tf.keras.applications.ResNet50(
        include_top = False,
        input_shape = input_shape
    )

    #ensure the model layers are frozen
    for layer in model.layers:
        layer.trainable = False

    return model

#function to return the DenseNet121 architecture as a base model
def build_densenet121_base(input_shape):
    model =  tf.keras.applications.DenseNet121(
        include_top = False,
        input_shape = input_shape
    )

    #ensure the model layers are frozen
    for layer in model.layers:
        layer.trainable = False

    return model

#function to create the custom head for the ResNet50 transfer learning model
def build_custom_head(base_model, K):

    hidden = Flatten()(base_model.output)
    hidden = Dense(200, activation='relu')(hidden)
    output_layer = Dense(K, activation = 'softmax')(hidden)

    return output_layer

#function to combine the base model and model head into a final model
def build_final_model(base_model, model_head):

    #build the final model out of the base model and model head
    final_model = Model(inputs = base_model.input, outputs = model_head)

    return final_model


###function to compile and train models
def compile_fit_model(model, train_data, validation_data, \
                      loss='sparse_categorical_crossentropy', optimizer='adam', \
                      metrics=['accuracy'], early_stop_patience=5, epochs=100):

    #compile the model
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics
    )

    #configure early stopping
    early_stop_callback = tf.keras.callbacks.EarlyStopping(
                              patience=early_stop_patience,
                              restore_best_weights=True
                          )

    #fit the model
    results = model.fit(
        train_data,
        validation_data = validation_data,
        epochs=epochs,
        callbacks=[early_stop_callback]
    )

    return results

##function to compile and evaluate the model on test data
def compile_evaluate_model(model, input_shape, weights_path, test_data, 
                         loss='sparse_categorical_crossentropy', 
                         optimizer='adam', metrics=['accuracy']):
    #build the model
    model.build(input_shape)

    #load the model weights
    model.load_weights(weights_path)

    #compile the model
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics
    )

    results = model.evaluate(test_data, verbose=1)

    return results


#function to plot the loss
def plot_loss(results):
    plt.plot(results.history['loss'], label='loss');
    plt.plot(results.history['val_loss'], label='val_loss');
    plt.legend();
    plt.gca().set_ylabel('Loss');
    plt.gca().set_xlabel('Epoch');
    plt.gca().set_title('Training and Validation Loss');   

    

#function to plot the accuracy
def plot_accuracy(results):
    plt.plot(results.history['accuracy'], label='acc');
    plt.plot(results.history['val_accuracy'], label='val_acc');
    plt.legend();  
    plt.gca().set_ylabel('Accuracy');
    plt.gca().set_xlabel('Epoch');
    plt.gca().set_title('Training and Validation Accuracy'); 
