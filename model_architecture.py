import keras
import tensorflow as tf
from keras import models
from keras import layers
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from utilss.TransformerClasses import *
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, TimeDistributed, GlobalMaxPooling2D, Dense
from tensorflow.keras.models import Model

def resnet_creator(num_classes, img_height, img_width):
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False,
                                                            input_shape=[img_height, img_width, 3])
    base_model.trainable = True
    model = keras.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(num_classes, activation="softmax"))
    return model
def resold():
    #input_tensor = Input(shape=(250, 250, 3))  # this assumes that your images have 3 channels (RGB)
    model = ResNet50(weights='imagenet', include_top=False)
    x = model.output
    x = layers.GlobalAveragePooling2D()(x)
    predictions = Dense(7, activation='softmax')(x)  # replace num_classes with your number of classes
    model = Model(inputs=model.input, outputs=predictions)
    return model
def get_sequence_model(sequence_length, embed_dim, dense_dim, num_heads, num_classes, img_height, img_width):
    input_shape = (img_height, img_width, 3)
    inputs = tf.keras.Input(shape=(sequence_length, *input_shape))
    resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
                      input_shape=input_shape, pooling=None)
    resnet.trainable = True
    x = TimeDistributed(resnet)(inputs)
    x = TimeDistributed(GlobalMaxPooling2D())(x)
    x = PositionalEmbedding(
        sequence_length, embed_dim, name="frame_position_embedding"
    )(x)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer1")(x)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer2")(x)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer3")(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model


"""def architecture_creator(num_classes, img_height, img_width):
    model = Sequential([
      layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
    ])
    return model"""

"""def architecture_creator(num_classes, img_height, img_width):
    backbone = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False)
    backbone.trainable = True
    inputs = layers.Input(shape=(img_height, img_width, 3))
    resize = layers.Resizing(32, 32)(inputs)
    neck = layers.Conv2D(3, (3,3), padding="same")(resize)
    preprocess_input = tf.keras.applications.mobilenet.preprocess_input(neck)
    x = backbone(preprocess_input)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs=inputs, outputs=outputs)"""

"""from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, Flatten, Dense, Add, Input
# Rest of the code remains the same as in the previous response
def residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=False):
    if conv_shortcut:
        shortcut = Conv2D(filters, 1, strides=stride)(x)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = x
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def architecture_creator(num_classes, img_height, img_width, num_blocks=[2, 2, 2]):
    input_tensor = Input(shape=(img_height, img_width, 3))
    x = Conv2D(64, 7, strides=2, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    for i, num_layers in enumerate(num_blocks):
        for j in range(num_layers):
            stride = 2 if j == 0 and i > 0 else 1
            x = residual_block(x, 64 * 2**i, stride=stride, conv_shortcut=True)
    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=output)
    return model"""



"""from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
def architecture_creator(num_classes, img_height, img_width):
    # Load ResNet50 with pre-trained weights on ImageNet and include the top layer
    base_model = ResNet50(weights='imagenet', include_top=True, input_shape=(img_height, img_width, 3))
    # Replace the last layer (1000 classes from ImageNet) with a Dense layer for your number of classes
    # Remove the last layer
    base_model.layers.pop()
    # Create a new model without the last layer
    model = models.Model(inputs=base_model.inputs, outputs=base_model.layers[-1].output)
    # Add new classification layer
    new_output = layers.Dense(num_classes, activation='softmax')(model.output)
    # Create a new model with the replaced output layer
    model = models.Model(inputs=model.inputs, outputs=new_output)
    return model"""

"""from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
def architecture_creator(num_classes, img_height, img_width):
    base_model = ResNet50(input_shape=(img_height, img_width, 3),
                          include_top=False,
                          weights=None)  # 'None' initializes random weights
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model"""



"""from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
def architecture_creator(num_classes, img_height, img_width):
    model = ResNet50(weights='imagenet', include_top=False)
    x = model.output
    x = layers.GlobalAveragePooling2D()(x)
    predictions = layers.Dense(num_classes, activation='softmax')(x)  # replace num_classes with your number of classes
    model = Model(inputs=model.input, outputs=predictions)
    return model"""

"""def architecture_creator(num_classes, img_height, img_width):
    backbone = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False)
    backbone.trainable = True
    inputs = layers.Input(shape=(img_height, img_width, 3))
    resize = layers.Resizing(32, 32)(inputs)
    neck = layers.Conv2D(3, (3,3), padding="same")(resize)
    preprocess_input = tf.keras.applications.mobilenet.preprocess_input(neck)
    x = backbone(preprocess_input)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs=inputs, outputs=outputs)
"""


