import os
import json
import glob
import matplotlib.pyplot as plt

def get_latest_log(work_dir):
    """Finds the most recent MMEngine scalars.json log file."""
    # MMEngine 3.x structure
    search_path = os.path.join(work_dir, '*', 'vis_data', 'scalars.json')
    log_files = glob.glob(search_path)
    
    if not log_files:
        # Fallback for alternative directory structures
        search_path = os.path.join(work_dir, '*.json')
        log_files = glob.glob(search_path)
        
    if not log_files:
        raise FileNotFoundError(f"Could not find any JSON log files in {work_dir}")
        
    # Return the file with the newest modification time
    newest_log = max(log_files, key=os.path.getmtime)
    return newest_log

def main():
    # work_dir = './work_dirs/custom_mask_rcnn'
    work_dir = './work_dirs/cascade_swin_small_v3'
    latest_log = get_latest_log(work_dir)
    print(f"[INFO] Parsing telemetry from: {latest_log}")

    steps = []
    losses = []
    val_epochs_raw = []
    val_ap50 = []

    # Parse the JSON lines
    with open(latest_log, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'loss' in data:
                    steps.append(data['step'])
                    losses.append(data['loss'])
                if 'coco/segm_mAP_50' in data:
                    val_epochs_raw.append(data['step']) 
                    val_ap50.append(data['coco/segm_mAP_50'])
            except json.JSONDecodeError:
                continue

    if not steps or not val_ap50:
        print("[ERROR] Could not find both loss and AP50 metrics in the log.")
        return

    # Dynamically align the Validation X-axis to the Training X-axis
    max_train_step = max(steps)
    max_val_epoch = max(val_epochs_raw)
    steps_per_epoch = max_train_step / max_val_epoch
    val_steps = [epoch * steps_per_epoch for epoch in val_epochs_raw]

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)

    # Plot Training Loss (Red)
    color1 = 'tab:red'
    ax1.set_xlabel('Training Steps', fontweight='bold')
    ax1.set_ylabel('Total Loss', color=color1, fontweight='bold')
    ax1.plot(steps, losses, color=color1, alpha=0.6, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Create dual Y-axis
    ax2 = ax1.twinx()  

    # Plot Validation AP50 (Blue)
    color2 = 'tab:blue'
    ax2.set_ylabel('Validation AP50', color=color2, fontweight='bold')
    ax2.plot(val_steps, val_ap50, color=color2, marker='o', linewidth=2, label='Validation AP50')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Formatting
    plt.title('Cascade Swin-S Training Convergence', fontsize=14, fontweight='bold')
    # plt.title('Cascade m_rcnn_r50 Training Convergence', fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Save the output
    output_filename = 'cascade_swin_small_v3.png'
    fig.tight_layout()
    plt.savefig(output_filename)
    print(f"[SUCCESS] Plot saved to {output_filename}")

if __name__ == '__main__':
    main()