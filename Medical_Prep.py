import os
import glob
import random
import pandas as pd
from PIL import Image

# --- 1. SETUP ---
CSV_FILE = './archive/csv/mass_case_description_train_set.csv'
BASE_IMAGE_DIR = './archive/jpeg'
OUTPUT_DIR = 'processed_medical_dataset'

folders = ["train/0_benign", "train/1_malignant", "test/0_benign", "test/1_malignant"]
for folder in folders:
    os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)

IMAGE_SIZE = (128, 128)
TRAIN_SPLIT = 0.8

random.seed(42)


# LOAD CSV

print("Reading CSV...")
df = pd.read_csv(CSV_FILE)
# Remove rows with missing pathology
df = df.dropna(subset=["pathology"])  # I think there is none but just in case


# MATCHING LOGIC

print("Matching CSV entries with JPEG folders...")
samples = []
for idx, row in df.iterrows():
    #try:
    path_parts = str(row["image file path"]).split("/")
    uid_folder = path_parts[2]  # SECOND UID is the key folder in JPEG structure
    folder_path = BASE_IMAGE_DIR + '/' + uid_folder

    if not os.path.exists(folder_path):
        continue

    pathology = str(row["pathology"]).strip().upper()
    if pathology == "MALIGNANT":
        label = "1_malignant"
    else:
        label = "0_benign"

    # collect ALL images in that folder
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
    for img_path in images:
        samples.append((img_path, label))

    #except Exception as e:
     #   continue

print(f"Matched {len(samples)} images with labels.")


# TRAIN / TEST SPLIT

random.shuffle(samples)
split_index = int(len(samples) * TRAIN_SPLIT)
train_samples = samples[:split_index]
test_samples = samples[split_index:]
print(f"Train samples: {len(train_samples)}")
print(f"Test samples : {len(test_samples)}")


# PROCESSING FUNCTION

def process_and_save(sample_list, subset_name):
    success_count = 0
    for index, (image_path, label) in enumerate(sample_list):

        try:
            img = Image.open(image_path)
            img = img.convert("L")
            img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)  #todo: to verif this
            save_name = f"{subset_name}_{index}.png"
            save_path = os.path.join(OUTPUT_DIR, subset_name, label, save_name)
            img.save(save_path)
            success_count += 1

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    return success_count


if __name__ == "__main__":
    print("\nProcessing training set...")
    train_count = process_and_save(train_samples, "train")

    print("Processing test set...")
    test_count = process_and_save(test_samples, "test")

    print("\n====================================")
    print(f"Training images saved : {train_count}")
    print(f"Testing images saved  : {test_count}")
    print("Dataset ready for PyTorch.")
    print("====================================")

"""
def prep_medical_dataset():
    print("Reading the CSV file to match labels...")
    # Load the CSV containing the hidden diagnoses
    df = pd.read_csv(CSV_FILE)
    
    success_count = 0
    
    print("Starting the resizing and sorting process. This might take a minute...")
    
    # Loop through every single row in the CSV
    for index, row in df.iterrows():
        # Get the file path and the diagnosis from the current row
        image_path = os.path.join(BASE_IMAGE_DIR, row['image file path'])
        pathology = row['pathology'].strip().upper()
        
        # --- 2. BINARY CLASSIFICATION LOGIC ---
        # Grouping BENIGN and BENIGN_WITHOUT_CALLBACK into '0_benign'
        if pathology == 'MALIGNANT':
            target_folder = MALIGNANT_DIR
        else:
            target_folder = BENIGN_DIR
            
        try:
            # --- 3. RESIZING LOGIC ---
            # Open the massive original image
            img = Image.open(image_path)
            
            # Convert to grayscale ('L') just in case some are weirdly formatted
            img = img.convert('L')
            
            # Shrink to 128x128 to save our RAM from exploding during training
            img_resized = img.resize((128, 128))
            #todo: to test img.resize((128,128), Image.Resampling.LANCZOS)
            
            # Create a clean, safe filename and save it to the correct folder
            safe_filename = f"scan_{index}.png"
            save_path = os.path.join(target_folder, safe_filename)
            img_resized.save(save_path)
            
            success_count += 1
            
        except Exception as e:
            # If an image is missing or corrupted, we just skip it and keep going
            print(f"Skipped image at {image_path}: {e}")

    print(f"\nDone! Successfully processed {success_count} images.")
    print(f"Check the '{OUTPUT_DIR}' folder. Your data is ready for PyTorch!")


if __name__ == "__main__":
    prep_medical_dataset()
"""
