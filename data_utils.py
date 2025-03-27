import os
import pandas as pd
import shutil
from PIL import Image

def classify_plant_pathology(base_dir, csv_file="train.csv", image_subdir="images"):
    """
    Classify leaf images into 'healthy' and 'sick' folders based on train.csv labels.
    Works with Plant Pathology dataset structure.
    """
    image_dir = os.path.join(base_dir, image_subdir)
    healthy_dir = os.path.join(base_dir, "healthy")
    sick_dir = os.path.join(base_dir, "sick")

    os.makedirs(healthy_dir, exist_ok=True)
    os.makedirs(sick_dir, exist_ok=True)

    csv_path = os.path.join(base_dir, csv_file)
    train_df = pd.read_csv(csv_path)

    for _, row in train_df.iterrows():
        image_name = row["image_id"] + ".jpg"
        image_path = os.path.join(image_dir, image_name)
        
        if not os.path.exists(image_path):
            continue

        if row["healthy"] == 1:
            shutil.move(image_path, os.path.join(healthy_dir, image_name))
        else:
            shutil.move(image_path, os.path.join(sick_dir, image_name))

    try:
        shutil.rmtree(image_dir)
        print(f"Removed original image folder: {image_dir}")
    except Exception as e:
        print(f"Failed to delete {image_dir}: {e}")

    print("Image classification completed (healthy vs sick).")


def resize_images_in_folder(input_folder, output_folder=None, size=(288, 288)):
    """
    Resize all images in input_folder to the specified size.
    If output_folder is given, save resized images there. Otherwise, overwrite.
    """
    if output_folder is None:
        output_folder = input_folder
    else:
        os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for fname in image_files:
        input_path = os.path.join(input_folder, fname)
        output_path = os.path.join(output_folder, fname)

        try:
            img = Image.open(input_path).convert("RGB")
            img_resized = img.resize(size)
            img_resized.save(output_path)
        except Exception as e:
            print(f"Failed to process {fname}: {e}")

    print(f"All images resized to {size} in '{output_folder}'.")