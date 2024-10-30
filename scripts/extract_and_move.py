import argparse
import os
import shutil
import tempfile
import zipfile
from tqdm import tqdm

def extract_and_move_wav(zip_path, target_folder):
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        for root, dirs, files in os.walk(temp_dir):
            wav_files = [f for f in files if f.endswith('.wav')]
            for wav_file in tqdm(wav_files, desc=f"Processing {os.path.basename(zip_path)}"):
                shutil.move(os.path.join(root, wav_file), os.path.join(target_folder, wav_file))

def main(data_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    zip_files = [f for f in os.listdir(data_path) if f.endswith('.zip')]
    for zip_file in tqdm(zip_files, desc="Overall Progress"):
        extract_and_move_wav(os.path.join(data_path, zip_file), output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and move WAV files from ZIP archives.")
    parser.add_argument("-d", "--data_path", required=True, help="Path to the directory containing ZIP files.")
    parser.add_argument("-o", "--output_path", required=True, help="Path to the directory where WAV files will be saved.")
    
    args = parser.parse_args()
    main(args.data_path, args.output_path)

