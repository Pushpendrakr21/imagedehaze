import tensorflow as tf
from tqdm import tqdm
from src.config import BATCH_SIZE, EPOCHS, CHECKPOINT_DIR
from src.generator import build_generator
from src.discriminator import build_discriminator
from src.losses import generator_loss, discriminator_loss
from src.data_loader import get_dataset

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Initialize datasets
train_dataset, val_dataset = get_dataset()

# Build models
generator = build_generator()
discriminator = build_discriminator()

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

# Checkpoints
checkpoint = tf.train.Checkpoint(generator=generator, discriminator=discriminator)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=5)

# Training loop
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    epoch_gen_loss = 0
    epoch_disc_loss = 0

    # Progress bar with tqdm
    for step, (hazy, clean) in enumerate(tqdm(train_dataset, desc=f"Epoch {epoch+1} Progress", unit="batch")):
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Forward pass
            generated_image = generator(hazy, training=True)

            # Calculate losses
            disc_real = discriminator([hazy, clean], training=True)
            disc_fake = discriminator([hazy, generated_image], training=True)

            gen_loss = generator_loss(disc_fake, generated_image, clean)
            disc_loss = discriminator_loss(disc_real, disc_fake)

        # Backpropagation
        grads_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
        grads_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        # Apply gradients
        generator_optimizer.apply_gradients(zip(grads_gen, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(grads_disc, discriminator.trainable_variables))

        # Accumulate loss for reporting
        epoch_gen_loss += gen_loss
        epoch_disc_loss += disc_loss

    # Print loss for the epoch
    print(f"Epoch {epoch+1} - Generator Loss: {epoch_gen_loss/step:.4f} - Discriminator Loss: {epoch_disc_loss/step:.4f}")

    # Save checkpoint
    checkpoint_manager.save()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# import os
# import tensorflow as tf
# from src.generator import build_generator
# from src.discriminator import build_discriminator
# from src.data_loader import get_dataset
# from src.losses import combined_loss
# from src.config import *

# cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# def discriminator_loss(real_output, fake_output):
#     real_loss = cross_entropy(tf.ones_like(real_output), real_output)
#     fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
#     return real_loss + fake_loss

# def generator_loss(fake_output, gen_output, target):
#     gan_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
#     combined = combined_loss(target, gen_output)
#     return gan_loss + combined

# generator = build_generator()
# discriminator = build_discriminator()

# gen_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.5)
# disc_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.5)

# train_dataset = get_dataset(TRAIN_HAZY_DIR, TRAIN_CLEAN_DIR)

# checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt")
# checkpoint = tf.train.Checkpoint(generator=generator, discriminator=discriminator)

# @tf.function
# def train_step(hazy, clean):
#     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#         gen_output = generator(hazy, training=True)

#         real_output = discriminator([hazy, clean], training=True)
#         fake_output = discriminator([hazy, gen_output], training=True)

#         gen_loss = generator_loss(fake_output, gen_output, clean)
#         disc_loss = discriminator_loss(real_output, fake_output)

#     gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
#     gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

#     gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
#     disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

#     return gen_loss, disc_loss

# def train(epochs):
#     for epoch in range(epochs):
#         print(f"\n🔁 Epoch {epoch + 1}/{epochs}")
#         for step, (hazy, clean) in enumerate(train_dataset):
#             gen_loss, disc_loss = train_step(hazy, clean)
#             if step % 10 == 0:
#                 print(f"Step {step}: Generator Loss = {gen_loss:.4f}, Discriminator Loss = {disc_loss:.4f}")
#         checkpoint.save(file_prefix=checkpoint_prefix)

# if __name__ == "__main__":
#     train(epochs=20)  # You can change this as needed
