import os
import shutil
import random

# Define source and destination directories
source_face_crop = "MSFD/1/face_crop"
dest_faces = "Input/masked_faces"

# Ensure destination directories exist
os.makedirs(dest_faces, exist_ok=True)
os.makedirs(dest_ground_truths, exist_ok=True)

# Get list of image files in the source face_crop directory
image_files = [f for f in os.listdir(source_face_crop) if os.path.isfile(os.path.join(source_face_crop, f))]

# Randomly select 100 images
selected_images = random.sample(image_files, 0)

# Copy images to their respective destinations
for image in selected_images:
    # Copy from face_crop to masked_faces
    shutil.copy(os.path.join(source_face_crop, image), os.path.join(dest_faces, image))

print("100 images and their corresponding segmentations copied successfully!")
