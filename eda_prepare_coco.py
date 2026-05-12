import os
import json
import numpy as np
from pycocotools import mask as maskUtils
from vrdl_dataset import MedicalCellDataset
import random
import warnings

warnings.filterwarnings("ignore")

def encode_mask_to_coco_rle(binary_mask):
    fortran_mask = np.asfortranarray(binary_mask)
    rle = maskUtils.encode(fortran_mask)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def build_coco_json(dataset, indices, output_json_path, start_img_id=0, start_ann_id=0):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "class1"},
            {"id": 2, "name": "class2"},
            {"id": 3, "name": "class3"},
            {"id": 4, "name": "class4"}
        ]
    }
    
    ann_id = start_ann_id
    print(f"[*] Compiling {len(indices)} images for {output_json_path}...")
    
    for idx in indices:
        image_tensor, target = dataset[idx]
        img_dir_name = dataset.image_dirs[idx]
        img_file_name = f"{img_dir_name}/image.tif" 
        
        height, width = image_tensor.shape[:2]
        img_id = start_img_id + idx
        
        coco_format["images"].append({
            "id": img_id,
            "file_name": img_file_name,
            "height": height,
            "width": width
        })
        
        masks = target["masks"].numpy()
        labels = target["labels"].numpy()
        
        for i in range(masks.shape[0]):
            binary_mask = masks[i]
            label_id = int(labels[i])
            
            rle = encode_mask_to_coco_rle(binary_mask)
            area = float(maskUtils.area(rle))
            bbox = maskUtils.toBbox(rle).tolist() 
            
            coco_format["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": label_id,
                "segmentation": rle,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            })
            ann_id += 1

    with open(output_json_path, 'w') as f:
        json.dump(coco_format, f)
    
    print(f"[SUCCESS] Wrote {len(coco_format['annotations'])} annotations to {output_json_path}")

def main():
    dataset_path = os.path.join('data', 'train')
    dataset = MedicalCellDataset(root_dir=dataset_path)
    
    print("\n==================================================")
    print("[EDA] COMMENCING EXPLORATORY DATA ANALYSIS")
    print("==================================================")
    
    image_stats = []
    global_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    
    # 1. Gather EDA Statistics directly from the raw masks
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        labels = target["labels"].numpy()
        
        counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for label in labels:
            counts[label] += 1
            global_counts[label] += 1
            
        image_stats.append({
            'idx': idx,
            'name': dataset.image_dirs[idx],
            'counts': counts,
            'total_rare': counts[3] + counts[4]
        })

    print(f"Total Dataset Images: {len(dataset)}")
    print(f"Global Instance Counts:")
    print(f"  - Class 1 (Common): {global_counts[1]}")
    print(f"  - Class 2 (Common): {global_counts[2]}")
    print(f"  - Class 3 (Rare):   {global_counts[3]}")
    print(f"  - Class 4 (Rare):   {global_counts[4]}")
    
    # 2. The Custom Stratified Split
    rare_images = [s for s in image_stats if s['total_rare'] > 0]
    common_images = [s for s in image_stats if s['total_rare'] == 0]
    
    print(f"\nImages containing Rare Cells (Class 3/4): {len(rare_images)}")
    print(f"Images containing only Common Cells: {len(common_images)}")
    
    # Shuffle deterministically
    random.seed(42)
    random.shuffle(rare_images)
    random.shuffle(common_images)
    
    # Strategy: Force 30% of rare-containing images into the Validation set
    # This guarantees our local mAP will heavily penalize bad rare-cell detection
    val_rare_count = int(len(rare_images) * 0.30)
    val_common_count = int(len(common_images) * 0.15)
    
    val_stats = rare_images[:val_rare_count] + common_images[:val_common_count]
    train_stats = rare_images[val_rare_count:] + common_images[val_common_count:]
    
    val_indices = [s['idx'] for s in val_stats]
    train_indices = [s['idx'] for s in train_stats]
    
    print("\n==================================================")
    print("[SPLIT] MACRO-BALANCED DATASET CREATED")
    print("==================================================")
    print(f"Train Images: {len(train_indices)}")
    print(f"Val Images:   {len(val_indices)}")
    
    val_class_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for s in val_stats:
        for k in range(1, 5):
            val_class_counts[k] += s['counts'][k]
            
    print(f"\nValidation Set Composition:")
    print(f"  - Class 1: {val_class_counts[1]}")
    print(f"  - Class 2: {val_class_counts[2]}")
    print(f"  - Class 3: {val_class_counts[3]}")
    print(f"  - Class 4: {val_class_counts[4]}")
    print("==================================================\n")

    # 3. Generate the Custom COCO JSONs
    build_coco_json(dataset, train_indices, "data/train_coco.json")
    build_coco_json(dataset, val_indices, "data/val_coco.json")

if __name__ == "__main__":
    main()