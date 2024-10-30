import argparse
import os
import shutil
from tqdm import tqdm

def categorize_wav_files(input_path, output_path):
    # Scene categories with underscore concatenated labels
    categories = {
        'indoor': ['airport', 'shopping_mall', 'metro_station'],
        'outdoor': ['street_pedestrian', 'public_square', 'street_traffic', 'urban_park'],
        'transportation': ['bus', 'tram', 'underground_metro']
    }

    # Create output directories
    for category in categories:
        os.makedirs(os.path.join(output_path, category), exist_ok=True)

    # Counters for each category
    counts = {category: 0 for category in categories}

    # Process files
    wav_files = [f for f in os.listdir(input_path) if f.endswith('.wav')]
    total_files = len(wav_files)
    for wav_file in tqdm(wav_files, desc="Processing WAV files"):
        # Parse the scene label
        scene_label = '_'.join(wav_file.split('-')[0].lower().split('_'))
        
        # Determine the category
        for category, scenes in categories.items():
            if scene_label in scenes:
                counts[category] += 1
                source_file = os.path.join(input_path, wav_file)
                dest_file = os.path.join(output_path, category, wav_file)
                shutil.copy(source_file, dest_file)
                break

    # Assertion to check if the total file count matches the sum of categorized files
    assert total_files == sum(counts.values()), "Mismatch in file counts between total and categorized files."

    return counts

def main(input_path, output_path):
    counts = categorize_wav_files(input_path, output_path)
    for category, count in counts.items():
        print(f"Number of WAV files in '{category}': {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Categorize and copy WAV files based on their scene labels.")
    parser.add_argument("-d", "--data_path", required=True, help="Path to the directory containing WAV files.")
    parser.add_argument("-o", "--output_path", required=True, help="Path to the directory where categorized WAV files will be saved.")
    
    args = parser.parse_args()
    main(args.data_path, args.output_path)

