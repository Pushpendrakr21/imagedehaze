# import tensorflow as tf
# from tensorflow.keras import layers

# def downsample(filters, size, apply_batchnorm=True):
#     initializer = tf.random_normal_initializer(0., 0.02)
#     block = tf.keras.Sequential()
#     block.add(
#         layers.Conv2D(filters, size, strides=2, padding='same',
#                       kernel_initializer=initializer, use_bias=not apply_batchnorm)
#     )
#     if apply_batchnorm:
#         block.add(layers.BatchNormalization())
#     block.add(layers.LeakyReLU())
#     return block

# def upsample(filters, size, apply_dropout=False):
#     initializer = tf.random_normal_initializer(0., 0.02)
#     block = tf.keras.Sequential()
#     block.add(
#         layers.Conv2DTranspose(filters, size, strides=2, padding='same',
#                                kernel_initializer=initializer, use_bias=False)
#     )
#     block.add(layers.BatchNormalization())
#     if apply_dropout:
#         block.add(layers.Dropout(0.5))
#     block.add(layers.ReLU())
#     return block

# def build_generator():
#     inputs = layers.Input(shape=[256, 256, 3])

#     # Encoder: Downsampling
#     down_stack = [
#         downsample(64, 4, apply_batchnorm=False),
#         downsample(128, 4),
#         downsample(256, 4),
#         downsample(512, 4),
#         downsample(512, 4),
#         downsample(512, 4),
#         downsample(512, 4),
#         downsample(512, 4),
#     ]

#     # Decoder: Upsampling
#     up_stack = [
#         upsample(512, 4, apply_dropout=True),
#         upsample(512, 4, apply_dropout=True),
#         upsample(512, 4, apply_dropout=True),
#         upsample(512, 4),
#         upsample(256, 4),
#         upsample(128, 4),
#         upsample(64, 4),
#     ]

#     initializer = tf.random_normal_initializer(0., 0.02)
#     last = layers.Conv2DTranspose(3, 4,
#                                    strides=2,
#                                    padding='same',
#                                    kernel_initializer=initializer,
#                                    activation='tanh')  # [-1, 1]

#     x = inputs
#     skips = []
#     for down in down_stack:
#         x = down(x)
#         skips.append(x)
#     skips = reversed(skips[:-1])

#     for up, skip in zip(up_stack, skips):
#         x = up(x)
#         x = layers.Concatenate()([x, skip])

#     x = last(x)

#     return tf.keras.Model(inputs=inputs, outputs=x)




#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import tensorflow as tf
from tensorflow.keras import layers

def residual_block(x, filters, size=3):
    shortcut = x
    x = layers.Conv2D(filters, size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    return x

def build_generator(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    e1 = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
    e1 = layers.ReLU()(e1)
    e2 = layers.Conv2D(128, 4, strides=2, padding='same')(e1)
    e2 = layers.BatchNormalization()(e2)
    e2 = layers.ReLU()(e2)
    e3 = layers.Conv2D(256, 4, strides=2, padding='same')(e2)
    e3 = layers.BatchNormalization()(e3)
    e3 = layers.ReLU()(e3)

    # Bottleneck with residual blocks
    b = residual_block(e3, 256)
    b = residual_block(b, 256)

    # Decoder
    d1 = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(b)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.ReLU()(d1)
    d1 = layers.Concatenate()([d1, e2])
    
    d2 = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(d1)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.ReLU()(d2)
    d2 = layers.Concatenate()([d2, e1])
    
    d3 = layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid')(d2)

    return tf.keras.Model(inputs, d3, name="Residual_UNet_Generator")
