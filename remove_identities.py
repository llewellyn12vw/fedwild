import pandas as pd
import numpy as np
import argparse
import os

def remove_identities(csv_file, num_identities_to_remove, output_file=None, random_seed=42):
    """
    Remove x number of identities from a CSV file and save the result.
    
    Args:
        csv_file: Path to input CSV file
        num_identities_to_remove: Number of identities to remove
        output_file: Output file path (optional, defaults to input_file_reduced.csv)
        random_seed: Random seed for reproducible removal
    """
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Check if 'identity' column exists
    if 'identity' not in df.columns:
        raise ValueError("CSV file must contain 'identity' column")
    
    # Get unique identities
    unique_identities = df['identity'].unique()
    total_identities = len(unique_identities)
    
    print(f"Total identities in file: {total_identities}")
    print(f"Identities to remove: {num_identities_to_remove}")
    
    if num_identities_to_remove >= total_identities:
        raise ValueError(f"Cannot remove {num_identities_to_remove} identities from {total_identities} total identities")
    
    # Randomly select identities to remove
    np.random.seed(random_seed)
    identities_to_remove = np.random.choice(unique_identities, size=num_identities_to_remove, replace=False)
    
    print(f"Removing identities: {identities_to_remove}")
    
    # Filter out selected identities
    df_filtered = df[~df['identity'].isin(identities_to_remove)]
    
    # Generate output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(csv_file)[0]
        extension = os.path.splitext(csv_file)[1]
        output_file = f"{base_name}_reduced{extension}"
    
    # Save filtered data
    df_filtered.to_csv(output_file, index=False)
    
    remaining_identities = len(df_filtered['identity'].unique())
    remaining_samples = len(df_filtered)
    
    print(f"Remaining identities: {remaining_identities}")
    print(f"Remaining samples: {remaining_samples}")
    print(f"Saved to: {output_file}")
    
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove x number of identities from a CSV file")
    parser.add_argument("csv_file", help="Path to input CSV file")
    parser.add_argument("num_identities", type=int, help="Number of identities to remove")
    parser.add_argument("-o", "--output",default='/home/wellvw12/baselines/train.csv', help="Output file path (optional)")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    remove_identities(args.csv_file, args.num_identities, args.output, args.seed)