_base_ = [
    'mmdet::_base_/models/cascade-mask-rcnn_r50_fpn.py',
    'mmdet::_base_/default_runtime.py'
]

# Force the Cascade Heads to load COCO intelligence, preventing random initialization
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth'

# --- 1. SWIN-S BACKBONE (UPGRADED) ---
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 18, 2], # <--- Upgraded to 18 blocks for Swin-Small
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3, # Heavy Regularization
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=True,       # Gradient Checkpointing enabled for VRAM safety
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    
    neck=dict(in_channels=[96, 192, 384, 768]),
    
    # --- 2. MICRO-ANCHORS & RPN ---
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2], # Forces (8, 16, 32, 64, 128) anchors
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64])),

    # --- 3. CASCADE HEADS WITH GIoU & CLEAN CROSS-ENTROPY MASK ---
    roi_head=dict(
        bbox_head=[
            dict(type='Shared2FCBBoxHead', num_classes=4, 
                 reg_decoded_bbox=True, loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(type='Shared2FCBBoxHead', num_classes=4, 
                 reg_decoded_bbox=True, loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(type='Shared2FCBBoxHead', num_classes=4, 
                 reg_decoded_bbox=True, loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
        ],
        mask_head=dict(num_classes=4)
    ),
    
    # --- 4. RPN CAPACITY OVERRIDES ---
    train_cfg=dict(
        rpn=dict(
            nms_pre=4000,      
            max_per_img=2000,  
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.03,    
            max_per_img=512))  
)

# --- 5. CLEAN DATASET PIPELINE (NO COPYPASTE) ---
dataset_type = 'CocoDataset'
data_root = 'data/'
metainfo = {
    'classes': ('class1', 'class2', 'class3', 'class4'),
    'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)]
}

train_pipeline = [
    dict(type='LoadTiffFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomResize', scale=[(1333, 640), (1333, 832)], keep_ratio=True),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='PhotoMetricDistortion', 
         brightness_delta=32, contrast_range=(0.5, 1.5), 
         saturation_range=(0.5, 1.5), hue_delta=18),
    dict(type='Pad', size_divisor=32),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadTiffFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(
    batch_size=1, 
    num_workers=4,
    dataset=dict(
        type=dataset_type, data_root=data_root, ann_file='train_coco.json',
        data_prefix=dict(img='train/'), metainfo=metainfo, pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1, num_workers=2,
    dataset=dict(
        type=dataset_type, data_root=data_root, ann_file='val_coco.json',
        data_prefix=dict(img='train/'), metainfo=metainfo, pipeline=test_pipeline))

test_dataloader = val_dataloader
val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'val_coco.json', metric=['bbox', 'segm'])
test_evaluator = val_evaluator

# --- 6. SCHEDULE FOR SWIN ---
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05), 
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0) # Protect the Swin backbone
        }),
    accumulative_counts=4) 

# Compressed to 50 Epochs to prevent overfitting on clean data
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', T_max=50, eta_min=1e-6, begin=0, end=50, by_epoch=True)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=2, save_best='coco/segm_mAP_50'))