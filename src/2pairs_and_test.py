import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from time import time
from scipy.io import loadmat

# Helper function to get subject ID from a filename
def parse_filename(filename):
    """
    Parses a filename to extract the subject ID.
    Assumes a format like 'sXXX_YY.mat' or 'XXX_Y_Z.mat'.
    
    Args:
        filename (str): The name of the file.
        
    Returns:
        str: The subject ID (e.g., '001') or None if the format is invalid.
    """
    try:
        basename = os.path.basename(filename)
        # Handle both 'sXXX' and 'XXX' formats
        subject_id = basename.split('_')[0].strip('s') 
        return subject_id
    except IndexError:
        print(f"Warning: Could not parse filename {filename}. Skipping.")
        return None

# Mock function for demonstration to make the script runnable.
# Replace with your actual implementation.
def shiftbits_ham(template, shifts):
    """
    Mock function to simulate bit shifting for Hamming distance calculation.
    """
    if shifts == 0:
        return template
    elif shifts > 0:
        return np.roll(template, shifts, axis=1)
    else:
        return np.roll(template, shifts, axis=1)

# Your provided HammingDistance function integrated here
def HammingDistance(template1, mask1, template2, mask2):
    """
    Calculate the Hamming distance between two iris templates.
    """
    hd = np.nan

    # Ensure templates and masks are of a numerical type for bitwise ops
    template1 = template1.astype(np.uint8)
    template2 = template2.astype(np.uint8)
    mask1 = mask1.astype(np.uint8)
    mask2 = mask2.astype(np.uint8)

    # Shifting template left and right, use the lowest Hamming distance
    for shifts in range(-8, 9):
        template1s = shiftbits_ham(template1, shifts)
        mask1s = shiftbits_ham(mask1, shifts)

        mask = np.logical_or(mask1s, mask2)
        
        # Invert the mask logic: 0 for valid bits, 1 for masked bits
        valid_mask = np.logical_not(mask)
        
        totalbits = np.sum(valid_mask)

        # Bitwise XOR operation to find differences
        C = np.logical_xor(template1s, template2)
        # Apply the combined mask to only count differences in valid bits
        bitsdiff = np.sum(np.logical_and(C, valid_mask))
        
        if totalbits == 0:
            hd1 = np.nan
        else:
            hd1 = bitsdiff / totalbits
            
        if np.isnan(hd) or hd1 < hd:
            hd = hd1

    return hd

def calculate_eer_far_frr_curves(genuine_distances, impostor_distances):
    if not genuine_distances or not impostor_distances:
        print("Error: Genuine or impostor distance lists are empty.")
        return None, None, None, None

    genuine_distances = np.array(sorted(genuine_distances))
    impostor_distances = np.array(sorted(impostor_distances))
    
    # Filter out nan values
    genuine_distances = genuine_distances[~np.isnan(genuine_distances)]
    impostor_distances = impostor_distances[~np.isnan(impostor_distances)]

    thresholds = np.linspace(0.0, 1.0, 1000)
    
    far_curve = []
    frr_curve = []
    
    total_genuine = len(genuine_distances)
    total_impostor = len(impostor_distances)
    
    for T in thresholds:
        false_accepts = np.sum(impostor_distances <= T)
        far = false_accepts / total_impostor
        far_curve.append(far)
        
        false_rejects = np.sum(genuine_distances > T)
        frr = false_rejects / total_genuine
        frr_curve.append(frr)
        
    # Find the EER
    eer_index = np.argmin(np.abs(np.array(far_curve) - np.array(frr_curve)))
    eer = (far_curve[eer_index] + frr_curve[eer_index]) / 2
    
    return thresholds, far_curve, frr_curve, eer

def plot_eer_curves(thresholds, far_curve, frr_curve, eer, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, far_curve, label='False Acceptance Rate (FAR)')
    plt.plot(thresholds, frr_curve, label='False Rejection Rate (FRR)')
    plt.plot(thresholds, np.ones(len(thresholds)) * eer, '--', color='red', label=f'EER = {eer:.4f}')
    
    plt.title('FAR and FRR Curves')
    plt.xlabel('Hamming Distance Threshold')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a file
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving plot to {output_path}")
    plt.savefig(output_path)
    plt.close() # Close the figure to free up memory

def plot_hamming_distribution(genuine_distances, impostor_distances, output_path):
    plt.figure(figsize=(10, 6))
    plt.hist(genuine_distances, bins=50, alpha=0.5, label='Genuine Pairs')
    plt.hist(impostor_distances, bins=50, alpha=0.5, label='Impostor Pairs')
    plt.title('Hamming Distance Distribution')
    plt.xlabel('Hamming Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a file
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving plot to {output_path}")
    plt.savefig(output_path)
    plt.close() # Close the figure to free up memory

def create_and_compare_pairs(all_templates_dict):
    """
    Creates genuine and impostor pairs and calculates the Hamming distance.
    
    Args:
        all_templates_dict (dict): A dictionary mapping subject ID to a list of (template, mask) pairs.
    
    Returns:
        tuple: A tuple of lists, (genuine_distances, impostor_distances).
    """
    genuine_distances = []
    impostor_distances = []
    
    # Get a list of all subjects
    subjects = sorted(list(all_templates_dict.keys()))
    
    # --- Create Genuine Pairs ---
    print("Creating and comparing genuine pairs...")
    genuine_pairs_printed = 0
    for subject_id in subjects:
        templates_for_subject = all_templates_dict[subject_id]
        
        # Create pairs of all unique combinations of templates for the same subject
        for i in range(len(templates_for_subject)):
            for j in range(i + 1, len(templates_for_subject)):
                temp1, mask1 = templates_for_subject[i]
                temp2, mask2 = templates_for_subject[j]
                
                if temp1 is not None and temp2 is not None:
                    # Use the new HammingDistance function
                    distance = HammingDistance(temp1, mask1, temp2, mask2)
                    genuine_distances.append(distance)
                    
                    if genuine_pairs_printed < 5:
                        # Print an example pair for verification
                        file1 = f"subject_{subject_id}_img{i}"
                        file2 = f"subject_{subject_id}_img{j}"
                        print(f"  Genuine pair: {file1} vs {file2} | Distance: {distance:.4f}")
                        genuine_pairs_printed += 1
                        
    # --- Create Impostor Pairs ---
    print("\nCreating and comparing impostor pairs...")
    # We'll create a random subset of impostor pairs to keep it manageable
    num_impostor_pairs = 2268  # Adjust this number as needed
    
    # Create a flattened list of all templates for random selection
    all_templates_list = [(s_id, temp_data) for s_id, temp_list in all_templates_dict.items() for temp_data in temp_list]
    
    impostor_pairs_printed = 0
    for _ in tqdm(range(num_impostor_pairs)):
        # Randomly select two templates from different subjects
        pair_data1 = random.choice(all_templates_list)
        pair_data2 = random.choice(all_templates_list)
        
        # Ensure they are from different subjects
        while pair_data1[0] == pair_data2[0]:
            pair_data2 = random.choice(all_templates_list)
            
        temp1, mask1 = pair_data1[1]
        temp2, mask2 = pair_data2[1]

        if temp1 is not None and temp2 is not None:
            # Use the new HammingDistance function
            distance = HammingDistance(temp1, mask1, temp2, mask2)
            impostor_distances.append(distance)
            
            if impostor_pairs_printed < 5:
                # Print an example pair for verification
                file1 = f"subject_{pair_data1[0]}"
                file2 = f"subject_{pair_data2[0]}"
                print(f"  Impostor pair: {file1} vs {file2} | Distance: {distance:.4f}")
                impostor_pairs_printed += 1
            
    return genuine_distances, impostor_distances


if __name__ == '__main__':
    # parsing args from the terminal
    parser = argparse.ArgumentParser(description="Iris Template Comparison Pipeline")
    parser.add_argument("--template_dir", type=str, default="./templates/DWT/features",
                        help="Directory where the feature templates are stored.")
    # Add a new argument for the output plot file
    parser.add_argument("--output_plot_dir", type=str, default=".",
                        help="Directory to save the output plots.")
    args = parser.parse_args()

    # Find all saved .mat files
    template_files = glob(os.path.join(args.template_dir, "*.mat"))
    n_templates = len(template_files)

    if n_templates == 0:
        print("No .mat template files found. Please run your feature extraction script first.")
    else:
        print(f"Found {n_templates} templates for comparison.")
        
        # --- Step 1: Load all templates into memory ---
        print("\nLoading templates...")
        all_templates_dict = {}
        for file in tqdm(template_files):
            subject_id = parse_filename(file)
            if subject_id:
                try:
                    mat_contents = loadmat(file)
                    template = mat_contents['template']
                    mask = mat_contents['mask']
                    
                    if subject_id not in all_templates_dict:
                        all_templates_dict[subject_id] = []
                    all_templates_dict[subject_id].append((template, mask))
                except Exception as e:
                    print(f"Error loading {file}: {e}")
        
        # --- Step 2: Create and compare pairs ---
        start_time = time()
        genuine_distances, impostor_distances = create_and_compare_pairs(all_templates_dict)
        end_time = time()
        
        # --- Step 3: Calculate EER, FAR, FRR curves ---
        thresholds, far_curve, frr_curve, eer = calculate_eer_far_frr_curves(genuine_distances, impostor_distances)
        
        # --- Step 4: Print summary statistics ---
        print("\n--- Summary ---")
        print(f"Total genuine comparisons: {len(genuine_distances)}")
        if genuine_distances:
            print(f"Average genuine distance: {np.nanmean(genuine_distances):.4f}")
        
        print(f"Total impostor comparisons: {len(impostor_distances)}")
        if impostor_distances:
            print(f"Average impostor distance: {np.nanmean(impostor_distances):.4f}")
            
        print(f"Calculated EER: {eer:.4f}")
        print(f'\nComparison time: {end_time - start_time:.2f} [s]\n')

        # --- Step 5: Plot the results ---
        if thresholds is not None:
            # Plot the EER curves
            eer_plot_path = os.path.join(args.output_plot_dir, "eer_far_frr_curves.png")
            plot_eer_curves(thresholds, far_curve, frr_curve, eer, eer_plot_path)

            # Plot the Hamming distance distribution
            hd_plot_path = os.path.join(args.output_plot_dir, "hamming_distance_distribution.png")
            plot_hamming_distribution(genuine_distances, impostor_distances, hd_plot_path)
