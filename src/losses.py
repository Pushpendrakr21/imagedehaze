import tensorflow as tf

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_obj(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_obj(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

def generator_loss(disc_generated_output, gen_output, target, LAMBDA=100):
    gan_loss = loss_obj(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# import tensorflow as tf
# from tensorflow.keras.applications import VGG19
# from tensorflow.keras.applications.vgg19 import preprocess_input
# from tensorflow.keras.models import Model
# from skimage.metrics import structural_similarity as skimage_ssim

# # Perceptual Loss
# vgg = VGG19(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
# vgg.trainable = False
# perceptual_layer = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)

# def perceptual_loss(y_true, y_pred):
#     y_true = preprocess_input(y_true * 255.0)
#     y_pred = preprocess_input(y_pred * 255.0)
#     return tf.reduce_mean(tf.abs(perceptual_layer(y_true) - perceptual_layer(y_pred)))

# def ssim_loss(y_true, y_pred):
#     return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# def l1_loss(y_true, y_pred):
#     return tf.reduce_mean(tf.abs(y_true - y_pred))

# def combined_loss(y_true, y_pred):
#     return (
#         0.6 * l1_loss(y_true, y_pred) +
#         0.2 * ssim_loss(y_true, y_pred) +
#         0.2 * perceptual_loss(y_true, y_pred)
#     )
