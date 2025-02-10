#!/usr/bin/env python3
import os
import sys
import imghdr
from joblib import Parallel, delayed
import argparse
from tqdm import tqdm


def process_file(filepath):
    """
    Process a single file: detect image type and rename by appending the appropriate extension,
    if the file currently has no extension.
    """
    # Check if the file already has an extension.
    base, ext = os.path.splitext(filepath)
    if ext:
        #print(f"Skipping {filepath}: already has extension {ext}")
        return

    # Detect image type using imghdr.
    image_type = imghdr.what(filepath)
    if image_type is None:
        #print(f"Skipping {filepath}: not recognized as an image file")
        # os.remove(filepath)
        return

    # Choose the proper extension (convert 'jpeg' to '.jpg').
    if image_type == 'jpeg':
        new_ext = '.jpg'
    else:
        new_ext = '.' + image_type

    # Build the new file name.
    new_filepath = filepath + new_ext

    # Avoid collisions: if a file with the new name already exists, add a counter.
    counter = 1
    candidate = new_filepath
    while os.path.exists(candidate):
        candidate = f"{filepath}_{counter}{new_ext}"
        counter += 1
    new_filepath = candidate

    # Rename the file.
    #print(f"Renaming {filepath} -> {new_filepath}")
    os.rename(filepath, new_filepath)

def rename_images_parallel(directory, n_jobs=-1):
    """
    Recursively scan the directory, and rename image files in parallel.
    
    Parameters:
      directory (str): Root directory to scan.
      n_jobs (int): Number of jobs to run in parallel (-1 uses all available processors).
    """
    # Build a list of file paths to process.
    filepaths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            filepaths.append(os.path.join(root, file))
    
    # Process files in parallel.
    Parallel(n_jobs=n_jobs)(delayed(process_file)(fp) for fp in tqdm(filepaths))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Restore image extensions.')
    parser.add_argument('--target_dir', type=str, help='Directory to scan for files.')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of parallel jobs.')
    args = parser.parse_args()
    print(f"Scanning directory {args.target_dir}...")
    rename_images_parallel(args.target_dir, args.n_jobs)
