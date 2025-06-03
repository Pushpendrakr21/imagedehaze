import tensorflow as tf
import os
from src.config import BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, DATA_DIR

def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0
    return image

def load_pair(hazy_path, clean_path):
    hazy_image = load_image(hazy_path)
    clean_image = load_image(clean_path)
    return hazy_image, clean_image

def get_dataset():
    train_hazy_dir = os.path.join(DATA_DIR, 'train', 'hazy')
    train_clean_dir = os.path.join(DATA_DIR, 'train', 'clean')

    val_hazy_dir = os.path.join(DATA_DIR, 'val', 'hazy')
    val_clean_dir = os.path.join(DATA_DIR, 'val', 'clean')

    train_hazy_files = sorted([os.path.join(train_hazy_dir, f) for f in os.listdir(train_hazy_dir)])
    train_clean_files = sorted([os.path.join(train_clean_dir, f) for f in os.listdir(train_clean_dir)])

    val_hazy_files = sorted([os.path.join(val_hazy_dir, f) for f in os.listdir(val_hazy_dir)])
    val_clean_files = sorted([os.path.join(val_clean_dir, f) for f in os.listdir(val_clean_dir)])

    train_ds = tf.data.Dataset.from_tensor_slices((train_hazy_files, train_clean_files))
    train_ds = train_ds.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(100).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_hazy_files, val_clean_files))
    val_ds = val_ds.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# import tensorflow as tf
# from src.config import IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, TRAIN_HAZY_DIR, TRAIN_CLEAN_DIR

# def load_image_pair(hazy_path, clean_path):
#     hazy = tf.io.read_file(hazy_path)
#     clean = tf.io.read_file(clean_path)
#     hazy = tf.image.decode_jpeg(hazy, channels=3)
#     clean = tf.image.decode_jpeg(clean, channels=3)

#     hazy = tf.image.resize(hazy, [IMG_HEIGHT, IMG_WIDTH])
#     clean = tf.image.resize(clean, [IMG_HEIGHT, IMG_WIDTH])

#     hazy = tf.cast(hazy, tf.float32) / 255.0
#     clean = tf.cast(clean, tf.float32) / 255.0

#     return hazy, clean

# def augment(hazy, clean):
#     if tf.random.uniform(()) > 0.5:
#         hazy = tf.image.flip_left_right(hazy)
#         clean = tf.image.flip_left_right(clean)
#     if tf.random.uniform(()) > 0.5:
#         hazy = tf.image.adjust_brightness(hazy, 0.1)
#     return hazy, clean

# def get_dataset(hazy_dir, clean_dir, training=True):
#     hazy_files = tf.data.Dataset.list_files(hazy_dir + "/*", shuffle=training)

#     # clean_files = tf.data.Dataset.list_files(clean_dir + "/*.jpg", shuffle=training)
#     clean_files = tf.data.Dataset.list_files(clean_dir + "/*", shuffle=training)


#     dataset = tf.data.Dataset.zip((hazy_files, clean_files))
#     dataset = dataset.map(lambda h, c: load_image_pair(h, c), num_parallel_calls=tf.data.AUTOTUNE)

#     if training:
#         dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

#     return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
