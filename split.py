import os
import shutil


def select_images(source_folder, dest_train,dest_test, retio_images):
    # Ensure the destination folder exists
    os.makedirs(dest_train, exist_ok=True)
    os.makedirs(dest_test, exist_ok=True)

    # Get a list of all files in the source folder
    all_files = [file for file in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, file))]
    split = int(retio_images * len(all_files))
    # Randomly select the specified number of files
    selected_train = all_files[:split]
    selected_test = all_files[split:]

    # Copy the selected files to the destination folder
    for file in selected_train:
        shutil.copy(os.path.join(source_folder, file), os.path.join(dest_train, file))
    
    for file in selected_test:
        shutil.copy(os.path.join(source_folder, file), os.path.join(dest_test, file))

# Source folders
dog_folder = "datasets/Dog"
cat_folder = "datasets/Cat"

# Destination folders
training_folder_dogs = "training_set/Dog"
training_folder_cats = "training_set/Cat"
test_folder_cats = "test_set/Cat"
test_folder_dogs = "test_set/Dog"

# Number of photos to select
retio_images = 0.8

# Process the folders
select_images(dog_folder, training_folder_dogs,test_folder_dogs, retio_images)
select_images(cat_folder, training_folder_cats,test_folder_cats, retio_images)

print("Random images have been copied to the new folders.")
