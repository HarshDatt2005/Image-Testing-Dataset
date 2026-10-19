# Housing Feature Extraction Pipeline

This repository contains an automated computer vision pipeline designed to download high-resolution housing datasets, map image files to Property IDs (PIDs), and run a deep learning model to extract architectural features and visible assets.

## Project Overview

The system uses a PyTorch EfficientNet-B7 model to evaluate housing images. It handles the end-to-end process: fetching large files from Google Drive, preparing the data mappings, and outputting a finalized Excel dataset containing categorical predictions and structural scores.

## Prerequisites

* Python 3.8+
* A configured virtual environment (`venv`)

## Installation & Setup

1. **Clone the repository and navigate into the project:**
   ```bash
   git clone <your-repository-url>
   cd <repository-folder>
   ```

2. **Activate your virtual environment:**
   * **Windows:** `venv\Scripts\activate`
   * **Mac/Linux:** `source venv/bin/activate`

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Execution Pipeline

The workflow is broken down into three automated scripts. They must be run in the following order:

### Step 1: Fetch Assets
Run the download script to retrieve the heavy files required for inference:

```bash
python download_assets.py
```
* **What it does:** Uses `gdown` to download the `Testing_Dataset_.zip` and `highres_housing_model.pth` directly from Google Drive.
* **Data Handling:** The script automatically extracts the images into the project directory and deletes the temporary `.zip` file to conserve disk space.

### Step 2: Generate Image Mappings
Run the mapping script to index the dataset:

```bash
python generate_mapping.py
```
* **What it does:** Scans the `Testing Dataset` directory for valid image files (`.jpg`, `.jpeg`, `.png`).
* **PID Extraction:** It cleanly handles multiple images per property by isolating the base PID (e.g., splitting `1D1FED49_1.jpg` to map to PID `1D1FED49`).
* **Output:** Generates `Mapped_Images.xlsx` containing the precise `PID` and `Image_Name` for the model to read.

### Step 3: Run Feature Extraction
Execute the main inference script:

```bash
python feature_extraction.py
```
* **What it does:** Loads the mapped images, applies a 600x600 black letterbox padding to preserve aspect ratios, and passes the tensors through the EfficientNet-B7 model.
* **Features Extracted:** Evaluates building context, housing category, stories, roof type, wall type, structural condition, floor material, and wall finish. It also detects visible assets with a probability threshold > 0.50 (e.g., ACs, trucks, two-wheelers).
* **Output:** Saves the final compiled data to `Mapped_Images_with_Features.xlsx`.