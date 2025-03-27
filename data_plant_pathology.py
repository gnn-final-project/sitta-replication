from data_utils import *

# Step 1: Classify
classify_plant_pathology("./data/plant_pathology")

# Step 2: Resize
resize_images_in_folder("./data/plant_pathology/healthy")
resize_images_in_folder("./data/plant_pathology/sick")