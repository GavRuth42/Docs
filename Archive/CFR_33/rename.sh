#!/usr/bin/env bash

# Loop through all files in the current directory that contain spaces
for file in *\ *; do
  # Only proceed if the file actually exists (and isn't a directory)
  if [ -f "$file" ]; then
    # Replace each space with an underscore
    new_name="${file// /_}"
    echo "Renaming '$file' to '$new_name'..."
    mv "$file" "$new_name"
  fi
done

