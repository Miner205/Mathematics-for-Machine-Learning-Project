import os
import pandas as pd
from PIL import Image

# --- 1. SETUP: Defining our folders ---
CSV_FILE = 'mass_case_description_train_set.csv'
BASE_IMAGE_DIR = 'original_medical_images' # Replace with the actual folder name your team downloads
OUTPUT_DIR = 'processed_medical_dataset'

# We create two folders for our binary classification
BENIGN_DIR = os.path.join(OUTPUT_DIR, '0_benign')
MALIGNANT_DIR = os.path.join(OUTPUT_DIR, '1_malignant')

# Make sure the folders exist before we start putting images in them
os.makedirs(BENIGN_DIR, exist_ok=True)
os.makedirs(MALIGNANT_DIR, exist_ok=True)

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
