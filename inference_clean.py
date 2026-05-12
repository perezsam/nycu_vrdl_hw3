import os
import json
import glob
import argparse
import numpy as np
import tifffile
import pycocotools.mask as mask_util

from mmdet.apis import init_detector, inference_detector
from mmdet.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

# --- MATCHING THE TRAIN.PY PIPELINE EXACTLY ---
@TRANSFORMS.register_module(force=True)
class LoadTiffFromFile(BaseTransform):
    def transform(self, results):
        img = tifffile.imread(results['img_path'])
        
        # Drop 4th channel (Alpha/Extra)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
            
        # Normalize to uint8
        if img.dtype != np.uint8:
            img = (255 * (img - np.min(img)) / (np.max(img) - np.min(img))).astype(np.uint8)
            
        # MMDetection expects BGR format internally. tifffile loads RGB. 
        img = img[:, :, ::-1]
        
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results
# ----------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run Standard Inference (No TTA)")
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to the .pth checkpoint')
    parser.add_argument('--out', default='test-results-notta.json', help='Output JSON name')
    parser.add_argument('--score_thr', type=float, default=0.05, help='Confidence threshold')
    args = parser.parse_args()

    test_json_path = 'data/test_image_name_to_ids.json' 
    test_img_dir = 'data/test_release/'            
    
    print(f"Loading mapping from {test_json_path}...")
    with open(test_json_path, 'r') as f:
        mapping_raw = json.load(f)
        
    if isinstance(mapping_raw, list):
        img_name_to_id = {entry["file_name"]: entry["id"] for entry in mapping_raw}
    else:
        img_name_to_id = mapping_raw

    print(f"Initializing model from {args.config} and {args.checkpoint}...")
    model = init_detector(args.config, args.checkpoint, device='cuda:0')

    image_paths = glob.glob(os.path.join(test_img_dir, '*.tif'))
    total_images = len(image_paths)
    print(f"Found {total_images} test images. Starting standard inference...")

    predictions = []

    for idx, img_path in enumerate(image_paths, 1):
        img_name = os.path.basename(img_path)
        img_id = img_name_to_id[img_name]

        result = inference_detector(model, img_path)
        
        valid_instances = result.pred_instances[result.pred_instances.scores > args.score_thr]

        for i in range(len(valid_instances)):
            bbox = valid_instances.bboxes[i].cpu().numpy()
            score = float(valid_instances.scores[i].cpu().numpy())
            label = int(valid_instances.labels[i].cpu().numpy())
            mask = valid_instances.masks[i].cpu().numpy()

            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1

            mask_uint8 = mask.astype(np.uint8)
            mask_fortran = np.asfortranarray(mask_uint8)
            rle = mask_util.encode(mask_fortran)
            rle['counts'] = rle['counts'].decode('utf-8') 

            category_id = label + 1 

            predictions.append({
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [round(float(x), 3) for x in [x1, y1, w, h]],
                "score": round(score, 5),
                "segmentation": rle
            })
            
        if idx % 10 == 0 or idx == total_images:
            print(f"Processed {idx}/{total_images} images...")

    with open(args.out, 'w') as f:
        json.dump(predictions, f)
    
    print(f"Done! Saved {len(predictions)} predictions to {args.out}.")

if __name__ == '__main__':
    main()