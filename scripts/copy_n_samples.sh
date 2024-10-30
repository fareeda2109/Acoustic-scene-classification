#!/bin/bash

# Get input parameters
dataset_dir="$1"
sample_size_per_class="$2"
output_dir="$3"

# Create a temporary directory to store the randomized samples
temp_dir=$(mktemp -d)

# Copy randomized samples from each class folder to the temporary directory
for class_dir in $(find "$dataset_dir" -type d); do
    mkdir -p "$output_dir/$class_dir"

    find "$class_dir" -type f -name "*.wav" | shuf -n "$sample_size_per_class" | while read -r filename; do
        cp "$filename" "$output_dir/$class_dir"
    done
done

# Process the randomized samples further (e.g., extract features, train a model)

# Remove the temporary directory
rm -rf "$temp_dir"
