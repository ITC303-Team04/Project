import os
import sys
from PIL import Image

# Function to delete an image if all pixels are white
def remove_blank(image_path):
  # Open image
  image = Image.open(image_path)
  # Get image size
  width, height = image.size
  # Get image pixels
  pixels = image.load()
  # Check if all pixels are white
  for x in range(width):
      for y in range(height):
          if pixels[x, y] != (255, 255, 255, 255):
              return False
  # Delete image
  os.remove(image_path)
  return True

base_dir = "base/img"
mask_dir = "masks/img"

# Traverse mask directory
for root, dirs, files in os.walk(sys.argv[1] + mask_dir):
  
  for file in files:
    print("Checking file: " + file)
    # Check if file is an image
    if file.endswith(".png") or file.endswith(".jpg"):
      # Get image path
      image_path = os.path.join(root, file)
      # Check if image is blank
      if remove_blank(image_path):
        print("Deleted: " + image_path)
        # Delete base image
        os.remove(image_path.replace(mask_dir, base_dir))
        print("Deleted: " + image_path.replace(mask_dir, base_dir))