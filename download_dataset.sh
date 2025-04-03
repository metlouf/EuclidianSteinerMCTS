#!/bin/bash

# URL of the directory containing the files
BASE_URL="https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/"

# Use wget to fetch the list of files and filter those starting with 'estein'
FILES=$(wget -q -O - "$BASE_URL" | grep -o 'estein[^"<>]*' | sort -u)

mkdir -p data

# Download each file
for FILE in $FILES; do
    echo "Downloading: ${BASE_URL}${FILE}"
    wget  "${BASE_URL}${FILE}" -O "data/$FILE"
done

echo "All estein* files downloaded successfully."
