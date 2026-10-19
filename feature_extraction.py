import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION
INPUT_EXCEL = "Mapped_Images.xlsx" 
IMAGE_FOLDER = "./Testing Dataset/" 
MODEL_WEIGHTS_PATH = "highres_housing_model.pth"
OUTPUT_EXCEL = "Mapped_Images_with_Features.xlsx"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running inference on: {DEVICE}")

# FEATURE DEFINITIONS & REVERSE MAPS
CATEGORICAL_COLS = [
    'Image_Context_Encoded', 'Housing_Category_Encoded', 'Ext_Stories_Encoded', 
    'Ext_Roof_Type_Encoded', 'Ext_Wall_Type_Encoded', 'Ext_Structural_Condition_Encoded', 
    'Int_Floor_Material_Encoded', 'Int_Wall_Finish_Encoded'
]
ASSET_COLS = ['Asset_AC', 'Asset_Tractor', 'Asset_Truck', 'Asset_Two-Wheeler', 'Asset_Four-Wheeler']

REVERSE_MAPS = {
    'Image_Context_Encoded': {
        0: 'Exterior_House_Building',
        1: 'Interior_Flat_or_Room',
        2: 'Invalid_or_Unviewable'
    },
    'Housing_Category_Encoded': {0: 'Kutcha', 1: 'Semi-Pucca', 2: 'Pucca', 3: 'Premium'},
    'Ext_Stories_Encoded': {0: 'One Storied', 1: 'Two Storied', 2: 'Three Storied', 3: 'Four Storied', 4: 'Five Storied', 5: 'Six Storied', 6: 'Seven Storied', 7: 'Eight Storied', 8: 'Nine Storied', 9: 'Ten Storied'},
    'Ext_Roof_Type_Encoded': {0: 'Thatch/Tarpaulin', 1: 'Corrugated Tin/Metal', 2: 'Khaprail', 3: 'Asbestos', 4: 'Concrete'},
    'Ext_Wall_Type_Encoded': {0: 'Mud/Makeshift', 1: 'Exposed Brick', 2: 'Finished Concrete/Plaster'},
    'Ext_Structural_Condition_Encoded': {0: 'Dilapidated', 1: 'Poor', 2: 'Average', 3: 'Excellent'},
    'Int_Floor_Material_Encoded': {0: 'Mud/Earth', 1: 'Cement/Concrete', 2: 'Tiles/Marble'},
    'Int_Wall_Finish_Encoded': {0: 'Mud', 1: 'Bare_Brick', 2: 'Raw_Plaster', 3: 'Painted', 4: 'Tiles'}
}

# Neural Network Output Sizing
cat_class_counts = {
    'Image_Context_Encoded': 2,
    'Housing_Category_Encoded': 4,
    'Ext_Stories_Encoded': 9,
    'Ext_Roof_Type_Encoded': 5,
    'Ext_Wall_Type_Encoded': 3,
    'Ext_Structural_Condition_Encoded': 4,
    'Int_Floor_Material_Encoded': 3,
    'Int_Wall_Finish_Encoded': 5
}
num_asset_classes = len(ASSET_COLS)

# MODEL ARCHITECTURE
class HighResHousingModel(nn.Module):
    def __init__(self, cat_class_counts, num_asset_classes):
        super(HighResHousingModel, self).__init__()
        base_model = models.efficientnet_b7(weights=None)
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        hidden_size = base_model.classifier[1].in_features
        self.dropout = nn.Dropout(p=0.5)

        self.cat_heads = nn.ModuleDict({
            col: nn.Linear(hidden_size, num_classes)
            for col, num_classes in cat_class_counts.items()
        })
        self.reg_head = nn.Linear(hidden_size, 1)
        self.asset_head = nn.Linear(hidden_size, num_asset_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        outputs = {col: head(x) for col, head in self.cat_heads.items()}
        outputs['Overall_Structural_Score'] = self.reg_head(x).squeeze(dim=-1)
        outputs['Visible_Assets'] = self.asset_head(x)
        return outputs

# Initializing and Loading Weights
model = HighResHousingModel(cat_class_counts, num_asset_classes).to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
except RuntimeError as e:
    print(f"Error loading weights: {e}")
    exit()
    
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# INFERENCE LOOPING WITH TQDM
print("Loading Excel file...")
df = pd.read_excel(INPUT_EXCEL)
results = []

for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Images", unit="img"):
    pid = str(row['PID'])
    
    # Read the exact filename from the new Mapped_Images Excel file
    img_name = str(row['Image_Name']) 
    img_path = os.path.join(IMAGE_FOLDER, img_name)
            
    row_dict = row.to_dict()
    
    if os.path.exists(img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            img = ImageOps.pad(img, (600, 600), color=(0, 0, 0))
            input_tensor = transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = model(input_tensor)

            raw_predictions = {}
            for col in CATEGORICAL_COLS:
                pred_idx = torch.argmax(outputs[col], dim=1).item()
                if col in REVERSE_MAPS and pred_idx in REVERSE_MAPS[col]:
                    raw_predictions[col] = REVERSE_MAPS[col][pred_idx]
                else:
                    raw_predictions[col] = f"Class Index {pred_idx}"

            # LOGIC
            context = raw_predictions.get('Image_Context_Encoded', '')
            if 'Exterior' in context:
                raw_predictions['Int_Floor_Material_Encoded'] = 'N/A'
                raw_predictions['Int_Wall_Finish_Encoded'] = 'N/A'
            elif 'Interior' in context:
                raw_predictions['Ext_Stories_Encoded'] = 'N/A'
                raw_predictions['Ext_Roof_Type_Encoded'] = 'N/A'
                raw_predictions['Ext_Wall_Type_Encoded'] = 'N/A'

            clean_preds = {}
            for col, pred_text in raw_predictions.items():
                clean_col_name = col.replace('_Encoded', '')
                clean_preds[clean_col_name] = pred_text

            score = outputs['Overall_Structural_Score'].item()
            
            asset_logits = outputs['Visible_Assets'][0]
            asset_probs = torch.sigmoid(asset_logits).cpu().numpy()
            detected_assets = []
            for asset_name, prob in zip(ASSET_COLS, asset_probs):
                clean_name = asset_name.replace('Asset_', '').replace('_', ' ')
                if prob > 0.50: 
                    detected_assets.append(clean_name)
                    
            # MAPPING
            row_dict['Image_Context'] = clean_preds.get('Image_Context', 'N/A')
            row_dict['Housing_Category'] = clean_preds.get('Housing_Category', 'N/A')
            row_dict['Ext_Stories'] = clean_preds.get('Ext_Stories', 'N/A')
            row_dict['Ext_Roof_Type'] = clean_preds.get('Ext_Roof_Type', 'N/A')
            row_dict['Ext_Wall_Type'] = clean_preds.get('Ext_Wall_Type', 'N/A')
            row_dict['Int_Floor_Material'] = clean_preds.get('Int_Floor_Material', 'N/A')
            row_dict['Int_Wall_Finish'] = clean_preds.get('Int_Wall_Finish', 'N/A')
            row_dict['Visible_Assets_For_Income'] = ", ".join(detected_assets) if detected_assets else "[]"
            row_dict['Overall_Structural_Score'] = round(score, 2)
            row_dict['Ext_Structural_Condition'] = clean_preds.get('Ext_Structural_Condition', 'N/A')

        except Exception as e:
            tqdm.write(f"Error processing image {img_name}: {e}")
            row_dict['Image_Context'] = "Error"
    else:
        tqdm.write(f"Image not found for PID: {img_name}")
        row_dict['Image_Context'] = "Image_Not_Found"

    results.append(row_dict)

# EXPORT FINAL DATASET
final_df = pd.DataFrame(results)

# EXACT format constraint
columns_order = [
    'PID', 'Image_Name', 'Image_Context', 
    'Housing_Category', 'Ext_Stories', 'Ext_Roof_Type', 'Ext_Wall_Type', 
    'Int_Floor_Material', 'Int_Wall_Finish', 'Visible_Assets_For_Income', 
    'Overall_Structural_Score', 'Ext_Structural_Condition'
]

final_cols = [col for col in columns_order if col in final_df.columns]
final_df = final_df[final_cols]

print(f"\nSaving features to {OUTPUT_EXCEL}...")
final_df.to_excel(OUTPUT_EXCEL, index=False)
print("Done!")