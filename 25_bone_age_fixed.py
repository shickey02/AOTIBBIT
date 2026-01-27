import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import pandas as pd
import numpy as np
import os
import glob
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import kagglehub

def clean_xray(img_path):
    """
    The 'Bone Filter': Removes tags, text, and noise.
    Keeps ONLY the largest connected object (The Hand/Arm).
    """
    # 1. Read as Grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    # 2. Threshold (Create a binary map)
    # Bones are bright, background is dark.
    # We use a low threshold (20) to capture all soft tissue + bone
    _, thresh = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
    
    # 3. Find Contours (Islands)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return Image.fromarray(img).convert('RGB')
    
    # 4. Find the Largest Island (The Hand)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 5. Create a Mask
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    # 6. Apply Mask (Everything outside the hand becomes pure black)
    cleaned = cv2.bitwise_and(img, img, mask=mask)
    
    # Convert back to PIL for PyTorch
    return Image.fromarray(cleaned).convert('RGB')

class CleanBoneDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['id']}.png")
        
        # USE THE CLEANING FUNCTION
        image = clean_xray(img_path)
        if image is None: # Fallback for bad files
             image = Image.new('RGB', (256, 256))
        
        if self.transform:
            image = self.transform(image)
            
        age = torch.tensor(row['boneage'] / 240.0, dtype=torch.float32)
        return image, age, row['id']

class TrueChronometer:
    def __init__(self):
        print("--- INITIALIZING 'TRUE' BONE CHRONOMETER (Tag-Blind) ---")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Model
        self.model = models.resnet50(weights='DEFAULT')
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3), # Increased dropout to force generalization
            nn.Linear(512, 1)
        )
        self.model = self.model.to(self.device)
        
        # Augmentation: Random Rotation is KEY here.
        # It prevents the AI from memorizing "Bone is always at pixel 100,100"
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomRotation(20), # Rotate +/- 20 degrees
            transforms.RandomHorizontalFlip(), # Flip hands (Left vs Right doesn't matter for age)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

    def run_training_cycle(self):
        # 1. Get Data
        print("   -> Locating Dataset...")
        path = kagglehub.dataset_download("kmader/rsna-bone-age")
        base_dir = os.path.join(path, "boneage-training-dataset", "boneage-training-dataset")
        if not os.path.exists(base_dir):
             base_dir = os.path.join(path, "boneage_training_dataset", "boneage_training_dataset")
        csv_path = os.path.join(path, "boneage-training-dataset.csv")
        
        df = pd.read_csv(csv_path).sample(2000, random_state=42).reset_index(drop=True)
        
        # 2. Setup Clean Dataset
        print("   -> Pre-processing X-rays (Masking Non-Bone Artifacts)...")
        # We split datasets with different transforms
        full_data = CleanBoneDataset(df, base_dir, transform=None)
        train_size = int(0.8 * len(full_data))
        test_size = len(full_data) - train_size
        
        # We need to manually split to apply different transforms
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        
        train_set = CleanBoneDataset(train_df, base_dir, transform=self.train_transform)
        test_set = CleanBoneDataset(test_df, base_dir, transform=self.val_transform)
        
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
        
        # 3. Train
        print("\n--- RETRAINING ON ISOLATED BONE GEOMETRY ---")
        optimizer = optim.Adam(self.model.parameters(), lr=0.0003)
        criterion = nn.MSELoss()
        
        for epoch in range(5):
            self.model.train()
            run_loss = 0.0
            
            for imgs, ages, _ in train_loader:
                imgs, ages = imgs.to(self.device), ages.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(imgs).squeeze()
                loss = criterion(outputs, ages)
                loss.backward()
                optimizer.step()
                run_loss += loss.item()
                
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, ages, _ in test_loader:
                    imgs, ages = imgs.to(self.device), ages.to(self.device)
                    outputs = self.model(imgs).squeeze()
                    val_loss += criterion(outputs, ages).item()
            
            mae_months = (val_loss / len(test_loader)) * 240
            print(f"   -> Epoch {epoch+1}: Val Error: {mae_months:.1f} months")

        # 4. Verify with Heatmaps
        self.audit_model(test_loader)

    def audit_model(self, loader):
        print("\n--- VERIFYING GAZE (GRAD-CAM) ---")
        target_layers = [self.model.layer4[-1]]
        cam = GradCAM(model=self.model, target_layers=target_layers)
        
        output_dir = "fixed_bone_audit"
        os.makedirs(output_dir, exist_ok=True)
        
        images, ages, ids = next(iter(loader))
        images = images.to(self.device)
        
        # Generate CAMs for first 5 images
        grayscale_cams = cam(input_tensor=images[:5], targets=None)
        
        for i in range(5):
            img_tensor = images[i].cpu()
            # Un-normalize for display
            img = img_tensor.permute(1, 2, 0).numpy()
            img = img * [0.229] + [0.485] # Approx un-norm (simplified for grayscale)
            img = np.clip(img, 0, 1)
            
            # Create Heatmap
            viz = show_cam_on_image(img, grayscale_cams[i, :], use_rgb=True)
            
            # Save
            cv2.imwrite(f"{output_dir}/audit_{ids[i]}.jpg", cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
            
        print(f"   -> Audit images saved to /{output_dir}")
        print("   -> CHECK: The heatmap should now be on the WRIST/FINGERS, not the corners.")

if __name__ == "__main__":
    bot = TrueChronometer()
    bot.run_training_cycle()