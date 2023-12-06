#!/bin/bash

# Download the zip file from Google Drive
url=$1

echo "downloading ..."
wget -O ../tmp/compressed.zip $url

echo "extracting ..."
# Extract the contents of the zip file
unzip ../tmp/compressed.zip -d ../tmp/extracted_files

echo "deleting old code ..."
# Delete the old code
rm -rf ../code

echo "moving new code ..."
# Move the new code to the code folder
mv ../tmp/extracted_files ../code

echo "deleting tmp files ..."
# Delete the tmp files
rm -rf ../tmp

echo "done"