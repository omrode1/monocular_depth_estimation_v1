import os
import numpy as np
import scipy.io
import shutil
from glob import glob

from tqdm import tqdm

# Paths
DEPTH_DIR = 'data/Train400Depth'
IMG_DIR = 'data'
OUT_IMG_DIR = 'data/paired/train/images'
OUT_DEPTH_DIR = 'data/paired/train/depth'

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_DEPTH_DIR, exist_ok=True)

def mat_to_img_name(mat_name):
    # Example: depth_sph_corr-060705-17.10.14-p-018t000.mat -> img-060705-17.10.14-p-018t000.jpg
    base = os.path.basename(mat_name)
    if base.startswith('depth_sph_corr-') and base.endswith('.mat'):
        core = base[len('depth_sph_corr-'):-len('.mat')]
        return f'img-{core}.jpg'
    return None

def process():
    mat_files = glob(os.path.join(DEPTH_DIR, 'depth_sph_corr-*.mat'))
    paired = 0
    for mat_path in tqdm(mat_files, desc='Processing pairs'):
        img_name = mat_to_img_name(mat_path)
        if not img_name:
            continue
        img_path = os.path.join(IMG_DIR, img_name)
        if not os.path.exists(img_path):
            continue
        # Load depth
        mat = scipy.io.loadmat(mat_path)
        grid = mat['Position3DGrid']
        depth = grid[:, :, 3].astype(np.float32)  # (H, W)
        # Save image
        out_img_path = os.path.join(OUT_IMG_DIR, img_name)
        shutil.copy(img_path, out_img_path)
        # Save depth
        out_depth_path = os.path.join(OUT_DEPTH_DIR, img_name.replace('.jpg', '.npy'))
        np.save(out_depth_path, depth)
        paired += 1
    print(f'Total pairs processed: {paired}')

if __name__ == '__main__':
    process() 