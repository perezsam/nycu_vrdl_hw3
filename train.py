import os
import argparse
import tifffile
import numpy as np
import cv2  # <--- This is the missing line
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

# ====================================================================
# HOTFIX V3: OpenCV Ultimate Compatibility Patch
# 1. Converts illegal dtypes (int64/bool) caused by CopyPaste math to float32.
# 2. Chunks arrays larger than 32 channels to bypass CV_CN_MAX limits.
# ====================================================================
original_cv2_resize = cv2.resize

def patched_cv2_resize(src, dsize, *args, **kwargs):
    if src is None or not isinstance(src, np.ndarray) or src.size == 0:
        return original_cv2_resize(src, dsize, *args, **kwargs)

    orig_dtype = src.dtype
    # OpenCV strictly supports: uint8, int8, uint16, int16, float32, float64
    supported_dtypes = (np.uint8, np.int8, np.uint16, np.int16, np.float32, np.float64)
    
    # 1. Dtype Protection
    if orig_dtype not in supported_dtypes:
        src = src.astype(np.float32)

    # 2. Channel Limit Protection
    if len(src.shape) == 3 and src.shape[2] > 32:
        chunks = []
        for i in range(0, src.shape[2], 32):
            chunk = src[:, :, i:i+32]
            resized = original_cv2_resize(chunk, dsize, *args, **kwargs)
            if len(resized.shape) == 2:
                resized = resized[:, :, np.newaxis]
            chunks.append(resized)
        res = np.concatenate(chunks, axis=-1)
    else:
        res = original_cv2_resize(src, dsize, *args, **kwargs)

    # 3. Restore Original Dtype
    if orig_dtype not in supported_dtypes:
        if orig_dtype == bool:
            res = res > 0.5
        else:
            res = np.round(res).astype(orig_dtype)

    return res

# Override the global OpenCV resize function
cv2.resize = patched_cv2_resize
# ====================================================================

# DEFENSIVE DATA INGESTION: Registered in the main script to avoid config parser crashes
@TRANSFORMS.register_module()
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

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--config', default='configs/custom_mask_rcnn.py', help='train config file path')
    parser.add_argument('--work-dir', default='./work_dirs/custom_mask_rcnn', help='the dir to save logs and models')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    print(f"[INFO] Loading configuration from {args.config}")
    cfg = Config.fromfile(args.config)
    
    cfg.work_dir = args.work_dir
    
    print("[INFO] Initializing MMEngine Runner...")
    runner = Runner.from_cfg(cfg)
    
    print("[INFO] Commencing Training Loop...")
    runner.train()

if __name__ == '__main__':
    main()