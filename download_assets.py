import gdown
import zipfile
import os

# --- Configuration ---
DATASET_URL = 'https://drive.google.com/file/d/1_6XqfKPfcx3AzSKWPbMPcGj05fEYk5NA/view?usp=sharing'
WEIGHTS_URL = 'https://drive.google.com/file/d/1PwrM8RsaNCiPU_zHWzMRgpImIHWLXHQz/view?usp=sharing'

# File names for the downloads
DATASET_ZIP = 'Testing_Dataset_.zip'
WEIGHTS_FILE = 'highres_housing_model.pth'

def setup_assets():
    print("Downloading dataset...")
    # fuzzy=True allows gdown to process the full sharing link
    gdown.download(url=DATASET_URL, output=DATASET_ZIP, quiet=False, fuzzy=True)
    
    print("\nDownloading model weights...")
    gdown.download(url=WEIGHTS_URL, output=WEIGHTS_FILE, quiet=False, fuzzy=True)

    # Extracting the dataset
    if os.path.exists(DATASET_ZIP):
        print(f"\nExtracting {DATASET_ZIP}...")
        with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
            # Since the zip already contains the "Testing Dataset" folder, this will correctly drop that folder right into your project.
            zip_ref.extractall(".") 
        
        # Cleaning up the zip file to save space
        os.remove(DATASET_ZIP)
        print("Extraction complete. Temporary zip file removed.")
    
    print("\nAll assets downloaded and ready to use!")

if __name__ == '__main__':
    setup_assets()