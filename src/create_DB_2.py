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
# Assuming you have the following functions defined in utils.extractandenconding
# and the main script's directory.
# from utils.extractandenconding import extractFeature, imread
# from YOUR_MODULE import segment, normalize, circlecoords

# parsing args from the terminal
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default="../CASIA1/*",
                    help="Directory of the dataset")
parser.add_argument("--template_dir", type=str, default="./templates/CASIA1/",
                    help="Destination of the features database and visualizations")
parser.add_argument("--number_cores", type=int, default=cpu_count(),
                    help="Number of cores used in the matching.")
args = parser.parse_args()

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
        basename = os.path.basename(file)
        
        # 1. Segment the eye image
        im = imread(file, 0)
        ciriris, cirpupil, imwithnoise = segment(im, use_multiprocess=False)
        
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
        # Make sure the image is in a viewable format (e.g., replace NaNs with 0)
        imwithnoise_vis = np.nan_to_num(imwithnoise, nan=0)
        Image.fromarray(imwithnoise_vis.astype(np.uint8)).save(imwithnoise_path)

        # 2. Normalize the segmented result
        arr_polar, arr_noise = normalize(imwithnoise, ciriris[1], ciriris[0], ciriris[2], 
                                         cirpupil[1], cirpupil[0], cirpupil[2], 20, 240)
                                         
        # Save the normalized iris texture
        normalized_path = os.path.join(args.template_dir, "normalized", f"normalized_{basename}")
        # The normalize function returns values between 0-1, so we need to scale it
        normalized_vis = (arr_polar * 255).astype(np.uint8)
        Image.fromarray(normalized_vis).save(normalized_path)
        
        # Save the normalized noise mask
        noise_mask_path = os.path.join(args.template_dir, "normalized", f"noise_mask_{basename}")
        # The noise mask is a boolean array, so scale to 0-255 for visibility
        noise_mask_vis = (arr_noise * 255).astype(np.uint8)
        Image.fromarray(noise_mask_vis).save(noise_mask_path)
        
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

files = glob(os.path.join(args.dataset_dir, "*_1_*.jpg"))
n_files = len(files)
print("N# of files for which we are extracting and visualizing features:", n_files)

# extracting features using multiple cores (number_cores)
pools = Pool(processes=args.number_cores)
for _ in tqdm(pools.imap_unordered(pool_func, files), total=n_files):
    pass
    
# total time
end = time()
print('\nTotal time: {} [s]\n'.format(end-start))
print('dummy commit')