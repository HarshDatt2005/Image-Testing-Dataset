import os
import pandas as pd

# Defining our dataset folder path
IMAGE_DIR = "Testing Dataset" 
OUTPUT_EXCEL = "Mapped_Images.xlsx" 

def create_image_mapping():
    data = []
    valid_extensions = ('.jpg', '.jpeg', '.png')

    if not os.path.exists(IMAGE_DIR):
        print(f"Error: The directory '{IMAGE_DIR}' does not exist.")
        return

    print(f"Scanning '{IMAGE_DIR}' for images...")
    
    for filename in os.listdir(IMAGE_DIR):
        if filename.lower().endswith(valid_extensions):
            # Removing the file extension
            base_name = os.path.splitext(filename)[0]
            
            # Isolating the PID by splitting at the underscore (if it exists)
            # e.g., '1D1FED49_1' -> '1D1FED49'
            # e.g., '1D1GDC24' -> '1D1GDC24'
            pid = base_name.split('_')[0]
            
            # Using 'PID' to match your existing model code's expected column name
            data.append({
                'PID': pid, 
                'Image_Name': filename
            })

    # Creating a DataFrame and saving it to Excel
    df = pd.DataFrame(data)
    df.to_excel(OUTPUT_EXCEL, index=False)
    
    print(f"Success! Created '{OUTPUT_EXCEL}' with {len(df)} image-to-PID mappings.")

if __name__ == '__main__':
    create_image_mapping()