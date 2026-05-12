import os
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile  # Replaces cv2 for robust biomedical TIFF parsing

class MedicalCellDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        
        # Filter for directories only to avoid hidden files (like .DS_Store)
        self.image_dirs = [
            d for d in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, d))
        ]
        
        self.mask_files = ['class1.tif', 'class2.tif', 'class3.tif', 'class4.tif']

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_dir_name = self.image_dirs[idx]
        img_dir_path = os.path.join(self.root_dir, img_dir_name)
        
        # 1. Load the primary image using tifffile
        img_path = os.path.join(img_dir_path, 'image.tif')
        image = tifffile.imread(img_path)

        # ---> ADD THE 4-CHANNEL SLICE HERE <---
        # Force 3-channel RGB (Drop the Alpha/Extra channel)
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        
        # Ensure image is in standard HWC uint8 format for MMDetection pipelines
        if image.dtype != np.uint8:
            # Normalize to 0-255 if it's a higher bit depth
            image = (255 * (image - np.min(image)) / (np.max(image) - np.min(image))).astype(np.uint8)
        
        # 2. Extract and compile the instance masks
        masks = []
        labels = []
        
        for class_idx, mask_name in enumerate(self.mask_files):
            mask_path = os.path.join(img_dir_path, mask_name)
            
            if os.path.exists(mask_path):
                # Load mask with tifffile
                mask = tifffile.imread(mask_path)
                
                # Extract unique instance IDs, explicitly ignoring the background (0)
                obj_ids = np.unique(mask)
                obj_ids = obj_ids[obj_ids > 0] 
                
                for obj_id in obj_ids:
                    # Create a binary mask for this specific instance
                    instance_mask = (mask == obj_id).astype(np.uint8)
                    masks.append(instance_mask)
                    # Class IDs: 1-indexed (1 to 4)
                    labels.append(class_idx + 1)
        
        # 3. Convert lists to PyTorch Tensors
        if len(masks) > 0:
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            masks = torch.empty((0, image.shape[0], image.shape[1]), dtype=torch.uint8)
            labels = torch.empty((0,), dtype=torch.int64)
            
        target = {}
        target["masks"] = masks
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target