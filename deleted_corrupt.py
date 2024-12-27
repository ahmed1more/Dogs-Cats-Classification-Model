import os
import PIL
from PIL import Image

def validate_images(folder):
    for root, _, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Check file integrity
            except (PIL.UnidentifiedImageError, IOError):
                print(f"Corrupt file detected and removed: {file_path}")
                os.remove(file_path)

validate_images('training_set')
validate_images('test_set')
