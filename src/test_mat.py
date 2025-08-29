import numpy as np
from scipy.io import loadmat
import os

def print_mat_contents(file_path):
    """
    Loads a .mat file and prints the full contents of its 'template' and 'mask'.

    Args:
        file_path (str): The path to the .mat file.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return

    try:
        # Load the .mat file
        mat_contents = loadmat(file_path)

        # Extract the template and mask
        template = mat_contents['template']
        mask = mat_contents['mask']

        # Print the full numpy arrays
        # The 'threshold' option ensures that a large array is not truncated
        np.set_printoptions(threshold=np.inf)
        
        print("--- Template Contents ---")
        print(template)
        print("\n--- Mask Contents ---")
        print(mask)
        
    except Exception as e:
        print(f"Error loading the .mat file: {e}")

if __name__ == '__main__':
    # You can change this path to point to any .mat file you want to check
    # For example, to check the file you mentioned:
    # file_path = "src/templates/CASIA1/features/001_1_1.jpg.mat"
    
    file_path = "./templates/DWT/features/001_1_1.jpg.mat"
    print_mat_contents(file_path)
