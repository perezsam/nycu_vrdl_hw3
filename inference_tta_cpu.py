import os
import json
import glob
import argparse
import numpy as np
import tifffile
import torch
import torchvision
import pycocotools.mask as mask_util

from mmdet.apis import init_detector, inference_detector
from mmdet.registry import TRANSFORMS

# --- REGISTRY INJECTION ---
@TRANSFORMS.register_module(force=True)
class LoadTiffFromFile:
    """Ensures init_detector can build the test pipeline metadata."""
    def __init__(self, **kwargs): pass
    def __call__(self, results):
        img = tifffile.imread(results['img_path'])
        if img.ndim == 2: img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] >= 4: img = img[:, :, :3]
        results['img'] = img.astype(np.uint8)
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', default='test-results.json')
    parser.add_argument('--score_thr', type=float, default=0.03)
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

    print("Initializing Swin-Cascade on CPU...")
    # Force the model to load onto the CPU
    model = init_detector(args.config, args.checkpoint, device='cpu')

    image_paths = glob.glob(os.path.join(test_img_dir, '*.tif'))
    total_images = len(image_paths)
    print(f"Starting 4x CPU-TTA inference on {total_images} images. This will take longer but avoids VRAM OOM.")

    predictions = []

    for idx, img_path in enumerate(image_paths, 1):
        img_name = os.path.basename(img_path)
        img_id = img_name_to_id[img_name]

        # Load RGB and convert to BGR for MMDetection
        img_orig = tifffile.imread(img_path)
        if img_orig.ndim == 3 and img_orig.shape[2] >= 4:
            img_orig = img_orig[:, :, :3]
        img_bgr = img_orig[:, :, ::-1].copy() # RGB to BGR
        H, W = img_bgr.shape[:2]

        results_list = []
        # 4x TTA Variants: (flip_h, flip_v)
        variants = [(False, False), (True, False), (False, True), (True, True)]
        
        for flip_h, flip_v in variants:
            img_var = img_bgr.copy()
            if flip_h: img_var = np.ascontiguousarray(np.fliplr(img_var))
            if flip_v: img_var = np.ascontiguousarray(np.flipud(img_var))
            
            # Forward pass on CPU
            res = inference_detector(model, img_var)
            
            boxes = res.pred_instances.bboxes.clone()
            scores = res.pred_instances.scores.clone()
            labels = res.pred_instances.labels.clone()
            masks = res.pred_instances.masks.clone()
            
            # Geometric Un-flipping
            if flip_h:
                boxes_f = boxes.clone()
                boxes_f[:, 0] = W - boxes[:, 2]
                boxes_f[:, 2] = W - boxes[:, 0]
                boxes = boxes_f
                masks = torch.flip(masks, dims=[2])
            if flip_v:
                boxes_f = boxes.clone()
                boxes_f[:, 1] = H - boxes[:, 3]
                boxes_f[:, 3] = H - boxes[:, 1]
                boxes = boxes_f
                masks = torch.flip(masks, dims=[1])
            
            results_list.append({'boxes': boxes, 'scores': scores, 'labels': labels, 'masks': masks})

        all_boxes = torch.cat([r['boxes'] for r in results_list])
        all_scores = torch.cat([r['scores'] for r in results_list])
        all_labels = torch.cat([r['labels'] for r in results_list])
        all_masks = torch.cat([r['masks'] for r in results_list])

        if len(all_boxes) > 0:
            # Per-class NMS to merge the 4 TTA branches
            keep = torchvision.ops.batched_nms(all_boxes, all_scores, all_labels, iou_threshold=0.45)
            all_boxes, all_scores, all_labels, all_masks = all_boxes[keep], all_scores[keep], all_labels[keep], all_masks[keep]

        valid_idx = all_scores > args.score_thr
        for i in np.where(valid_idx.cpu().numpy())[0]:
            bbox = all_boxes[i].cpu().numpy()
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            
            mask_fortran = np.asfortranarray(all_masks[i].cpu().numpy().astype(np.uint8))
            rle = mask_util.encode(mask_fortran)
            rle['counts'] = rle['counts'].decode('utf-8')

            predictions.append({
                "image_id": img_id,
                "category_id": int(all_labels[i].cpu().numpy()) + 1,
                "bbox": [round(float(x), 3) for x in [x1, y1, w, h]],
                "score": round(float(all_scores[i].cpu().numpy()), 5),
                "segmentation": rle
            })
            
        if idx % 10 == 0 or idx == total_images:
            print(f"Processed {idx}/{total_images} images...")

    with open(args.out, 'w') as f:
        json.dump(predictions, f)
    print(f"SUCCESS: Saved {len(predictions)} predictions to {args.out}.")

if __name__ == '__main__':
    main()