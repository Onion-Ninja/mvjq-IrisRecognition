import argparse
import os
import numpy as np
from glob import glob
from tqdm import tqdm
from time import time
from scipy.io import savemat
from multiprocessing import cpu_count, Pool
from PIL import Image, ImageDraw
from utils.imgutils import segment, normalize
from utils.extractandenconding import encode_iris
import pywt

# parsing args from the terminal
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default="../CASIA1/*",
                    help="Directory of the dataset")
parser.add_argument("--template_dir", type=str, default="./templates/DWT/",
                    help="Destination of the features database and visualizations")
parser.add_argument("--number_cores", type=int, default=cpu_count(),
                    help="Number of cores used in the matching.")
args = parser.parse_args()

def encode_dwt(arr_polar, arr_noise):
    """
    Generate iris templates using 2D Discrete Wavelet Transform
    """
    coeffs = pywt.dwt2(arr_polar, 'haar')
    LL , (LH, HL, HH) = coeffs

    coeffs_noise = pywt.dwt2(arr_noise, 'haar')
    LL_n, (LH_n, HL_n, HH_n) = coeffs_noise

    
    template_lh = (LH > 0).astype(np.uint8)
    template_hl = (HL > 0).astype(np.uint8)
    template_hh = (HH > 0).astype(np.uint8)

    template = np.concatenate((template_lh, template_hl, template_hh), axis=1)
    
    mask_noise_lh = (LH_n > 0)
    mask_noise_hl = (HL_n > 0)
    mask_noise_hh = (HH_n > 0)
    
    mask_noise = np.concatenate((mask_noise_lh, mask_noise_hl, mask_noise_hh), axis=1)

    template_magnitude = np.concatenate((np.abs(LH), np.abs(HL), np.abs(HH)), axis=1)
    low_magnitude_mask = (template_magnitude < 0.0001)

    mask = np.logical_or(mask_noise, low_magnitude_mask).astype(np.uint8)

    return template, mask

def imread(filename, mode=0):
    img = np.array(Image.open(filename).convert('L'))
    return img



# creating a pool function to use with multiprocessing
def pool_func(file):
    try:
        # Original feature extraction (you can keep this or remove it if you only need visualization)
        # template, mask, _ = extractFeature(file, multiprocess=False)
        # out_file = os.path.join(args.template_dir, "%s.mat" % (os.path.basename(file)))
        # savemat(out_file, mdict={'template': template, 'mask': mask})

        # --- VISUALIZATION CODE ---
        # 1. Segment the eye image (no re-computation)
        im = imread(file, 0)
        eyelashes_threshold = 80
        multiprocess = False
        ciriris, cirpupil, imwithnoise = segment(im, eyelashes_threshold, multiprocess)
        
        # 2. Normalize the segmented result (no re-computation)
        radial_resolution = 20
        angular_resolution = 240
        arr_polar, arr_noise = normalize(imwithnoise, ciriris[1],  ciriris[0], ciriris[2], 
                                         cirpupil[1], cirpupil[0], cirpupil[2],
                                         radial_resolution, angular_resolution)
        
        # 3. Feature encoding (no re-computation)
        # minw_length = 18
        # mult = 1
        # sigma_f = 0.5
        template, mask = encode_dwt(arr_polar, arr_noise)

        # 4. Save the generated template and mask
        basename = os.path.basename(file)
        features_dir = os.path.join(args.template_dir, "features")
        os.makedirs(features_dir, exist_ok=True)
        out_file = os.path.join(features_dir, "%s.mat" % (os.path.basename(file)))
        savemat(out_file, mdict={'template': template, 'mask': mask})

        """
        Visualization not needed as only feature encoding changes
        # --- VISUALIZATION CODE (now uses the variables created above) ---
        # Draw the circles on the original image for visualization
        im_segmented = Image.fromarray(im).convert("RGB")
        draw = ImageDraw.Draw(im_segmented)
        
        # Draw Iris Circle
        draw.ellipse([ciriris[1] - ciriris[2], ciriris[0] - ciriris[2],
                      ciriris[1] + ciriris[2], ciriris[0] + ciriris[2]],
                      outline='red', width=2)
        
        # Draw Pupil Circle
        draw.ellipse([cirpupil[1] - cirpupil[2], cirpupil[0] - cirpupil[2],
                      cirpupil[1] + cirpupil[2], cirpupil[0] + cirpupil[2]],
                      outline='blue', width=2)
                      
        # Save the segmented result
        segmented_path = os.path.join(args.template_dir, "segmented", f"segmented_{basename}")
        im_segmented.save(segmented_path)
        
        # Save the noise-masked image
        imwithnoise_path = os.path.join(args.template_dir, "segmented", f"with_noise_{basename}")
        imwithnoise_vis = np.nan_to_num(imwithnoise, nan=0)
        Image.fromarray(imwithnoise_vis.astype(np.uint8)).save(imwithnoise_path)

        # Save the normalized iris texture
        normalized_path = os.path.join(args.template_dir, "normalized", f"normalized_{basename}")
        normalized_vis = (arr_polar * 255).astype(np.uint8)
        Image.fromarray(normalized_vis).save(normalized_path)
        
        # Save the normalized noise mask
        noise_mask_path = os.path.join(args.template_dir, "normalized", f"noise_mask_{basename}")
        noise_mask_vis = (arr_noise * 255).astype(np.uint8)
        Image.fromarray(noise_mask_vis).save(noise_mask_path)
        """
    except Exception as e:
        print(f"Error processing file {file}: {e}")

# time it
start = time()
if not os.path.exists(args.template_dir):
    print("makedirs", args.template_dir)
    os.makedirs(args.template_dir)

# Create subdirectories for visualizations
os.makedirs(os.path.join(args.template_dir, "segmented"), exist_ok=True)
os.makedirs(os.path.join(args.template_dir, "normalized"), exist_ok=True)

files = glob(os.path.join(args.dataset_dir, "*_*.jpg"))
n_files = len(files)
print("N# of files for which we are extracting and visualizing features:", n_files)

# extracting features using multiple cores (number_cores)
pools = Pool(processes=args.number_cores)
for _ in tqdm(pools.imap_unordered(pool_func, files), total=n_files):
    pass
    
# total time
end = time()
print('\nTotal time: {} [s]\n'.format(end-start))
# print('dummy commit')