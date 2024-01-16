import tensorflow as tf

def split_all_data(data_dir,batch_size,img_height,img_width,validation_split,seed):
    train_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=validation_split,
      subset="training",
      seed=seed,
      image_size=(img_height, img_width),
      batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=validation_split,
      subset="validation",
      seed=seed,
      image_size=(img_height, img_width),
      batch_size=batch_size)
    return (train_ds, val_ds)