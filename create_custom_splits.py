import os
import random

def create_splits(base_dir):
    # CULane uses train_gt.txt which contains lines with:
    # <img_path> <mask_path> <4_lane_exist_binary_flags>
    train_gt_path = os.path.join(base_dir, 'list', 'train_gt.txt')
    
    # We will output new gt files
    out_gt_5k_path  = os.path.join(base_dir, 'list', 'train_gt_5k.txt')
    out_gt_10k_path = os.path.join(base_dir, 'list', 'train_gt_10k.txt')
    
    # We should also probably output normal train_5k.txt and train_10k.txt (just img paths)
    out_5k_path  = os.path.join(base_dir, 'list', 'train_5k.txt')
    out_10k_path = os.path.join(base_dir, 'list', 'train_10k.txt')
    
    # 1. Read all training ground truths
    with open(train_gt_path, 'r') as f:
        all_lines = f.readlines()
        
    print(f"Total lines in train_gt.txt: {len(all_lines)}")
    
    # 2. Filter lines that belong to the specifically requested driver directories
    # 1. Training&validation images and annotations:
    # driver_23_30frame
    # driver_161_90frame
    # driver_182_30frame
    target_dirs = ['driver_23_30frame', 'driver_161_90frame', 'driver_182_30frame']
    
    # Stratified collection to ensure equal distribution from the three targets
    dir_lines = {t: [] for t in target_dirs}
    
    for line in all_lines:
        for tdir in target_dirs:
            if tdir in line:
                dir_lines[tdir].append(line)
                break # A line belongs to at most one dir
                
    for tdir in target_dirs:
        print(f"Lines found for {tdir}: {len(dir_lines[tdir])}")
        
    # Check minimum length for uniform sampling
    min_count = min(len(lines) for lines in dir_lines.values())
    print(f"Minimum lines available in a single directory: {min_count}")
    
    # Shuffle each dir independently
    random.seed(42)
    for tdir in target_dirs:
        random.shuffle(dir_lines[tdir])
        
    # 3. Create 5k split (Need ~1666 from each)
    target_5k = 5000
    per_dir_5k = target_5k // len(target_dirs)
    rem_5k = target_5k % len(target_dirs)
    
    lines_5k = []
    for i, tdir in enumerate(target_dirs):
        count = per_dir_5k + (1 if i < rem_5k else 0)
        lines_5k.extend(dir_lines[tdir][:count])
        
    # 4. Create 10k split (Need ~3333 from each)
    target_10k = 10000
    per_dir_10k = target_10k // len(target_dirs)
    rem_10k = target_10k % len(target_dirs)
    
    lines_10k = []
    for i, tdir in enumerate(target_dirs):
        count = per_dir_10k + (1 if i < rem_10k else 0)
        lines_10k.extend(dir_lines[tdir][:count])
        
    # Shuffle the final lists so batches are mixed
    random.shuffle(lines_5k)
    random.shuffle(lines_10k)
    
    print(f"\nFinal 5k split size: {len(lines_5k)}")
    print(f"Final 10k split size: {len(lines_10k)}")
    
    # 5. Function to extract just the img paths for `train_XXX.txt`
    def write_splits(lines, gt_out_path, img_out_path):
        with open(gt_out_path, 'w') as f_gt, open(img_out_path, 'w') as f_img:
            for line in lines:
                f_gt.write(line)
                # the image path is the first element
                img_path = line.split()[0]
                f_img.write(f"{img_path}\n")
    
    write_splits(lines_5k, out_gt_5k_path, out_5k_path)
    write_splits(lines_10k, out_gt_10k_path, out_10k_path)
    
    print(f"\nSaved:")
    print(f" - {out_gt_5k_path}")
    print(f" - {out_5k_path}")
    print(f" - {out_gt_10k_path}")
    print(f" - {out_10k_path}")


if __name__ == '__main__':
    base_dir = r"\\wsl.localhost\Ubuntu-24.04\home\alki\projects\CULane"
    create_splits(base_dir)
