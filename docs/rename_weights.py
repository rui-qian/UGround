#!/usr/bin/env python3
"""
Script to rename weight keys from 'gmp' to 'ppm' in PyTorch model files.
This script will update both the .bin files and the index.json files.
"""

import os
import json
import torch
import argparse
from pathlib import Path

def rename_keys_in_state_dict(state_dict):
    """Rename keys from gmp.* to ppm.* in state dict"""
    new_state_dict = {}
    renamed_count = 0
    
    for key, value in state_dict.items():
        if key.startswith('gmp.'):
            new_key = key.replace('gmp.', 'ppm.', 1)
            new_state_dict[new_key] = value
            renamed_count += 1
            print(f"  Renamed: {key} -> {new_key}")
        else:
            new_state_dict[key] = value
    
    return new_state_dict, renamed_count

def update_index_json(index_path):
    """Update the pytorch_model.bin.index.json file"""
    print(f"Updating index file: {index_path}")
    
    with open(index_path, 'r') as f:
        index_data = json.load(f)
    
    # Update weight_map keys
    if 'weight_map' in index_data:
        new_weight_map = {}
        renamed_count = 0
        
        for key, value in index_data['weight_map'].items():
            if key.startswith('gmp.'):
                new_key = key.replace('gmp.', 'ppm.', 1)
                new_weight_map[new_key] = value
                renamed_count += 1
                print(f"  Index renamed: {key} -> {new_key}")
            else:
                new_weight_map[key] = value
        
        index_data['weight_map'] = new_weight_map
        
        # Save updated index
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        print(f"  Updated {renamed_count} keys in index file")
        return renamed_count
    
    return 0

def process_model_directory(model_dir):
    """Process a single model directory"""
    model_path = Path(model_dir)
    if not model_path.exists():
        print(f"Directory does not exist: {model_dir}")
        return False
    
    print(f"\nProcessing directory: {model_dir}")
    
    # Find all .bin files
    bin_files = list(model_path.glob("pytorch_model*.bin"))
    if not bin_files:
        print("  No .bin files found")
        return False
    
    # Process each .bin file
    total_renamed = 0
    for bin_file in bin_files:
        print(f"\nProcessing: {bin_file}")
        
        try:
            # Load the state dict
            state_dict = torch.load(bin_file, map_location='cpu')
            
            # Rename keys
            new_state_dict, renamed_count = rename_keys_in_state_dict(state_dict)
            
            if renamed_count > 0:
                # Save the updated state dict
                torch.save(new_state_dict, bin_file)
                print(f"  Saved {renamed_count} renamed keys to {bin_file}")
                total_renamed += renamed_count
            else:
                print(f"  No keys to rename in {bin_file}")
                
        except Exception as e:
            print(f"  Error processing {bin_file}: {e}")
            return False
    
    # Update index.json file
    index_file = model_path / "pytorch_model.bin.index.json"
    if index_file.exists():
        try:
            index_renamed = update_index_json(index_file)
            total_renamed += index_renamed
        except Exception as e:
            print(f"  Error updating index file: {e}")
            return False
    else:
        print("  No index.json file found")
    
    print(f"\nCompleted processing {model_dir}")
    print(f"Total keys renamed: {total_renamed}")
    return total_renamed > 0

def main():
    parser = argparse.ArgumentParser(description='Rename gmp.* keys to ppm.* in PyTorch model files')
    parser.add_argument('model_dirs', nargs='+', help='Model directories to process')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be renamed without making changes')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
    
    success_count = 0
    for model_dir in args.model_dirs:
        if process_model_directory(model_dir):
            success_count += 1
    
    print(f"\nSummary: Successfully processed {success_count}/{len(args.model_dirs)} directories")

if __name__ == "__main__":
    main()
