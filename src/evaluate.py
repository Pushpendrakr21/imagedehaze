# import os
# import numpy as np
# import tensorflow as tf
# import cv2
# import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as ssim

# from src.config import VAL_HAZY_DIR, VAL_CLEAN_DIR, OUTPUT_DIR, IMG_HEIGHT, IMG_WIDTH
# from src.generator import build_generator  # Make sure this function exists and returns your generator model


# # Load the generator model from checkpoint
# def load_generator(checkpoint_dir):
#     generator = build_generator()  # Must match architecture used in training
#     checkpoint = tf.train.Checkpoint(generator=generator)
#     checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
#     print(f"✅ Generator restored from: {tf.train.latest_checkpoint(checkpoint_dir)}")
#     return generator


# # Calculate PSNR
# def psnr_metric(y_true, y_pred):
#     max_pixel = 1.0
#     mse = np.mean((y_true - y_pred) ** 2)
#     return 20 * np.log10(max_pixel / np.sqrt(mse))


# # Save and display output images
# def save_and_display_output(predicted_image, clean_image, output_path, image_name):
#     os.makedirs(output_path, exist_ok=True)
#     output_image_path = os.path.join(output_path, f"predicted_{image_name}")
#     cv2.imwrite(output_image_path, cv2.cvtColor((predicted_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.title("Ground Truth (Clean)")
#     plt.imshow(clean_image)
#     plt.axis("off")

#     plt.subplot(1, 2, 2)
#     plt.title("Predicted (Dehazed)")
#     plt.imshow(predicted_image)
#     plt.axis("off")
#     plt.show()


# # Evaluation function
# def evaluate(generator, hazy_dir, clean_dir, output_dir):
#     psnr_values = []
#     ssim_values = []

#     for image_name in os.listdir(hazy_dir):
#         hazy_path = os.path.join(hazy_dir, image_name)
#         clean_path = os.path.join(clean_dir, image_name)

#         hazy_image = cv2.imread(hazy_path)
#         clean_image = cv2.imread(clean_path)

#         hazy_image = cv2.cvtColor(hazy_image, cv2.COLOR_BGR2RGB) / 255.0
#         clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB) / 255.0

#         hazy_image = cv2.resize(hazy_image, (IMG_WIDTH, IMG_HEIGHT))
#         clean_image = cv2.resize(clean_image, (IMG_WIDTH, IMG_HEIGHT))

#         pred_image = generator(np.expand_dims(hazy_image, axis=0), training=False)[0].numpy()
#         pred_image = np.clip(pred_image, 0.0, 1.0)

#         psnr_value = psnr_metric(clean_image, pred_image)
#         ssim_value = ssim(clean_image, pred_image, channel_axis=-1, data_range=1.0)


#         psnr_values.append(psnr_value)
#         ssim_values.append(ssim_value)

#         print(f"{image_name} | PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}")
#         save_and_display_output(pred_image, clean_image, output_dir, image_name)

#     print(f"\n📊 Average PSNR: {np.mean(psnr_values):.2f}")
#     print(f"📊 Average SSIM: {np.mean(ssim_values):.4f}")


# # Main execution
# def main():
#     checkpoint_dir = 'checkpoints'  # just folder, not ckpt file
#     generator = load_generator(checkpoint_dir)
#     evaluate(generator, VAL_HAZY_DIR, VAL_CLEAN_DIR, OUTPUT_DIR)

# if __name__ == "__main__":
#     main()
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import numpy as np
import tensorflow as tf
import cv2
from skimage.metrics import structural_similarity as ssim

from src.config import VAL_HAZY_DIR, VAL_CLEAN_DIR, OUTPUT_DIR, IMG_HEIGHT, IMG_WIDTH
from src.generator import build_generator


# Load the generator model from checkpoint
def load_generator(checkpoint_dir):
    generator = build_generator()
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    print(f"✅ Generator restored from: {tf.train.latest_checkpoint(checkpoint_dir)}")
    return generator


# PSNR calculation
def psnr_metric(y_true, y_pred):
    max_pixel = 1.0
    mse = np.mean((y_true - y_pred) ** 2)
    return 20 * np.log10(max_pixel / np.sqrt(mse))


# Save hazy, clean, and predicted as a single side-by-side image
def save_combined_output(hazy, clean, predicted, output_path, image_name):
    os.makedirs(output_path, exist_ok=True)
    combined = np.hstack([
        (hazy * 255).astype(np.uint8),
        (clean * 255).astype(np.uint8),
        (predicted * 255).astype(np.uint8)
    ])
    combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_path, f"combined_{image_name}"), combined)


# Evaluate the model
def evaluate(generator, hazy_dir, clean_dir, output_dir):
    psnr_values = []
    ssim_values = []

    for image_name in os.listdir(hazy_dir):
        hazy_path = os.path.join(hazy_dir, image_name)

        # Replace 'hazy' with 'GT' to get the clean image name
        clean_image_name = image_name.replace("hazy", "GT")
        clean_path = os.path.join(clean_dir, clean_image_name)

        # Read hazy and clean images
        hazy_img = cv2.imread(hazy_path)
        clean_img = cv2.imread(clean_path)

        # Skip if images are not read properly
        if hazy_img is None:
            print(f"❌ Failed to read hazy image: {hazy_path}")
            continue
        if clean_img is None:
            print(f"❌ Failed to read clean image: {clean_path}")
            continue

        # Preprocess
        hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB) / 255.0
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB) / 255.0

        hazy_img = cv2.resize(hazy_img, (IMG_WIDTH, IMG_HEIGHT))
        clean_img = cv2.resize(clean_img, (IMG_WIDTH, IMG_HEIGHT))

        # Generate prediction
        pred_img = generator(np.expand_dims(hazy_img, axis=0), training=False)[0].numpy()
        pred_img = np.clip(pred_img, 0.0, 1.0)

        # Compute metrics
        psnr = psnr_metric(clean_img, pred_img)
        ssim_val = ssim(clean_img, pred_img, channel_axis=-1, data_range=1.0)

        psnr_values.append(psnr)
        ssim_values.append(ssim_val)

        print(f"{image_name} | PSNR: {psnr:.2f}, SSIM: {ssim_val:.4f}")
        save_combined_output(hazy_img, clean_img, pred_img, output_dir, image_name)

    print(f"\n📊 Average PSNR: {np.mean(psnr_values):.2f}")
    print(f"📊 Average SSIM: {np.mean(ssim_values):.4f}")


# Main runner
def main():
    checkpoint_dir = 'checkpoints'
    generator = load_generator(checkpoint_dir)
    evaluate(generator, VAL_HAZY_DIR, VAL_CLEAN_DIR, OUTPUT_DIR)

if __name__ == "__main__":
    main()


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# import os
# import numpy as np
# import tensorflow as tf
# import cv2
# from tqdm import tqdm
# from src.config import VAL_HAZY_DIR, VAL_CLEAN_DIR, OUTPUT_DIR, IMG_HEIGHT, IMG_WIDTH
# from src.generator import build_generator

# # Load generator from latest checkpoint
# def load_generator(checkpoint_dir):
#     generator = build_generator()
#     checkpoint = tf.train.Checkpoint(generator=generator)
#     latest = tf.train.latest_checkpoint(checkpoint_dir)
#     if latest:
#         checkpoint.restore(latest).expect_partial()
#         print(f"✅ Generator restored from: {latest}")
#     else:
#         raise FileNotFoundError("❌ No checkpoint found!")
#     return generator

# # Preprocess image
# def load_and_preprocess(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Image at {image_path} could not be read.")
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
#     image = image.astype(np.float32) / 255.0
#     return image

# # Save predicted image
# def save_output(image, output_path, name):
#     os.makedirs(output_path, exist_ok=True)
#     save_path = os.path.join(output_path, f"predicted_{name}")
#     cv2.imwrite(save_path, cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

# # Evaluate function
# def evaluate(generator, hazy_dir, clean_dir, output_dir):
#     psnr_scores = []
#     ssim_scores = []

#     image_files = [f for f in os.listdir(hazy_dir) if f.lower().endswith(('.jpg', '.png'))]

#     for img_name in tqdm(image_files, desc="Evaluating"):
#         hazy_img = load_and_preprocess(os.path.join(hazy_dir, img_name))
#         clean_img = load_and_preprocess(os.path.join(clean_dir, img_name))

#         hazy_tensor = tf.expand_dims(hazy_img, axis=0)
#         pred_tensor = generator(hazy_tensor, training=False)
#         pred_image = tf.squeeze(pred_tensor, axis=0).numpy()
#         pred_image = np.clip(pred_image, 0.0, 1.0)

#         # Compute PSNR and SSIM
#         psnr_val = tf.image.psnr(pred_image, clean_img, max_val=1.0).numpy()
#         ssim_val = tf.image.ssim(pred_image, clean_img, max_val=1.0).numpy()

#         psnr_scores.append(psnr_val)
#         ssim_scores.append(ssim_val)

#         save_output(pred_image, output_dir, img_name)

#     print(f"\n📊 Average PSNR: {np.mean(psnr_scores):.2f}")
#     print(f"📊 Average SSIM: {np.mean(ssim_scores):.4f}")

# # Entry point
# def main():
#     checkpoint_dir = "checkpoints"
#     generator = load_generator(checkpoint_dir)
#     evaluate(generator, VAL_HAZY_DIR, VAL_CLEAN_DIR, OUTPUT_DIR)

# if __name__ == "__main__":
#     main()
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# import os
# import numpy as np
# import tensorflow as tf
# import cv2
# import csv
# from skimage.metrics import structural_similarity as ssim
# import matplotlib.pyplot as plt

# from src.config import VAL_HAZY_DIR, VAL_CLEAN_DIR, OUTPUT_DIR, IMG_HEIGHT, IMG_WIDTH
# from src.generator import build_generator


# def load_generator(checkpoint_dir):
#     generator = build_generator()
#     checkpoint = tf.train.Checkpoint(generator=generator)
#     checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
#     print(f"✅ Generator restored from: {tf.train.latest_checkpoint(checkpoint_dir)}")
#     return generator


# def psnr_metric(y_true, y_pred):
#     max_pixel = 1.0
#     mse = np.mean((y_true - y_pred) ** 2)
#     return 20 * np.log10(max_pixel / np.sqrt(mse))


# def save_side_by_side(predicted_image, clean_image, output_path, image_name):
#     os.makedirs(output_path, exist_ok=True)
#     plt.figure(figsize=(10, 4))

#     plt.subplot(1, 2, 1)
#     plt.title("Ground Truth (Clean)")
#     plt.imshow(clean_image)
#     plt.axis("off")

#     plt.subplot(1, 2, 2)
#     plt.title("Predicted (Dehazed)")
#     plt.imshow(predicted_image)
#     plt.axis("off")

#     save_path = os.path.join(output_path, f"compare_{image_name}.png")
#     plt.savefig(save_path, bbox_inches='tight')
#     plt.close()


# def evaluate(generator, hazy_dir, clean_dir, output_dir, visualize=False, export_csv=True):
#     psnr_values = []
#     ssim_values = []

#     os.makedirs(output_dir, exist_ok=True)
#     csv_path = os.path.join(output_dir, "metrics.csv")

#     with open(csv_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["Image Name", "PSNR", "SSIM"])

#         for image_name in os.listdir(hazy_dir):
#             hazy_path = os.path.join(hazy_dir, image_name)
#             clean_path = os.path.join(clean_dir, image_name)

#             if not os.path.exists(clean_path):
#                 print(f"⚠️ Skipping {image_name} — clean image not found.")
#                 continue

#             hazy_image = cv2.imread(hazy_path)
#             clean_image = cv2.imread(clean_path)

#             hazy_image = cv2.cvtColor(hazy_image, cv2.COLOR_BGR2RGB) / 255.0
#             clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB) / 255.0

#             hazy_image = cv2.resize(hazy_image, (IMG_WIDTH, IMG_HEIGHT))
#             clean_image = cv2.resize(clean_image, (IMG_WIDTH, IMG_HEIGHT))

#             pred_image = generator(np.expand_dims(hazy_image, axis=0), training=False)[0].numpy()
#             pred_image = np.clip(pred_image, 0.0, 1.0)

#             psnr_value = psnr_metric(clean_image, pred_image)
#             ssim_value = ssim(clean_image, pred_image, channel_axis=-1, data_range=1.0)

#             psnr_values.append(psnr_value)
#             ssim_values.append(ssim_value)

#             writer.writerow([image_name, f"{psnr_value:.2f}", f"{ssim_value:.4f}"])
#             print(f"{image_name} | PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}")

#             # Save side-by-side comparison (optional)
#             if visualize:
#                 save_side_by_side(pred_image, clean_image, output_dir, os.path.splitext(image_name)[0])

#     print(f"\n📊 Average PSNR: {np.mean(psnr_values):.2f}")
#     print(f"📊 Average SSIM: {np.mean(ssim_values):.4f}")
#     if export_csv:
#         print(f"📁 Metrics saved to: {csv_path}")


# def main():
#     checkpoint_dir = 'checkpoints'
#     generator = load_generator(checkpoint_dir)

#     # Change visualize=True if you want to see/save comparison images
#     evaluate(generator, VAL_HAZY_DIR, VAL_CLEAN_DIR, OUTPUT_DIR, visualize=False)


# if __name__ == "__main__":
#     main()
