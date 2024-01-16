# standard imports
import os
import numpy as np
import random
from pathlib import Path
# custom imports
from dataloading.FrameGenerator import *
from utilss.Cholec_path_label_list import *
from model_architecture import *
from utilss.custom_data_split import *
from keras.callbacks import Callback
from sklearn.metrics import classification_report
# keras imports
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
# wandb imports
import wandb
from wandb.keras import WandbCallback


def main():
    # Weights and Biases configues
    wandb.init(project='cholec80_keras', entity='ali-baharimalayeri')
    config = wandb.config
    config.learning_rate = 0.00005
    config.batch_size = 64
    config.dataset_split_percentage = 50
    config.epochs = 25

    # define split share and number of classes
    num_classes = 7
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # This will hide the warnings
    tf.get_logger().setLevel('ERROR')  # This will hide the warnings but show errors
    wandb.login

    # Dataset Path
    data_dir = Path("E:\\Ali\\December2023\\video_frames_224\\")
    path_list, images_path_sep_folders, label_list_sep_folders = path_label_list(data_dir)
    if len(images_path_sep_folders) == len(label_list_sep_folders): numfolders = len(images_path_sep_folders)

    # Split the data
    train_data, validation_data = custom_data_split(numfolders, images_path_sep_folders, label_list_sep_folders, config.dataset_split_percentage)

    # Initialize dictionaries
    partition = {'train': [], 'validation': []}
    labels = {}

    # Populate the dictionaries
    for item in train_data:
        partition['train'].append(item[0])
        labels[item[0]] = item[1]
    for item in validation_data:
        partition['validation'].append(item[0])
        labels[item[0]] = item[1]

    # Dataset inpection
    image_count = len(list(data_dir.glob('*/*.jpg')))
    num_folers = len(list(data_dir.glob('*')))
    folder_names = [folder.name for folder in data_dir.glob('*') if folder.is_dir()]
    print("number of all images:\n", image_count)
    print("number of folders:\n", num_folers)
    print("name of folders:\n", folder_names, "\n\n")

    # Parameters
    params = {'dim': (224,224),
              'batch_size': config.batch_size,
              'n_classes': num_classes,
              'n_channels': 3,
              'shuffle': True}

    # Generators
    training_generator = FrameGenerator(partition['train'], labels, augmentation = True, **params)
    validation_generator = FrameGenerator(partition['validation'], labels, augmentation = False, **params)

    #Model Implementation
    # model = architecture_creator(num_classes, img_height, img_width)
    model = resnet_creator(num_classes, 224, 224)
    model.summary()

    # compile the model
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(optimizer=adam_optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    class CustomCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            temp_batch_pred = []
            temp_batch_label = []
            for images_batch, labels_batch in validation_generator:
                temp_batch_label.extend(labels_batch)
                predicted = model.predict(images_batch, verbose=0)
                predicted_classes = np.argmax(predicted, axis=1)
                temp_batch_pred.extend(predicted_classes)
                #print(classification_report(labels_batch, predicted_classes))
            print(classification_report(temp_batch_label, temp_batch_pred))

    #Training Procedure
    model.fit(training_generator,
              validation_data=validation_generator,
              epochs=config.epochs,
              #callbacks=[CustomCallback())
              callbacks=[WandbCallback(save_model=False), ModelCheckpoint(filepath="check_points\\{epoch:02d}-{val_accuracy:.2f}.keras", save_best_only=True)])
              #callbacks=[CustomCallback(), ModelCheckpoint("{epoch:02d}-{val_loss:.2f}.keras", save_best_only=True)])
    wandb.finish()

if __name__ == '__main__':
    main()
