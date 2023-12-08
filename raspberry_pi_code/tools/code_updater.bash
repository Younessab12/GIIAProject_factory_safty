#!/bin/bash

# Download the zip file from Google Drive
absolute_url="" #$1

# extract download id
# ex https://drive.google.com/file/d/1MyDe5__ItWjyzSaNaAHR93yyDTKkoMgt/view?usp=drive_link
# to 1MyDe5__ItWjyzSaNaAHR93yyDTKkoMgt
download_id=$(echo $absolute_url | cut -d'/' -f6)
download_id="1MyDe5__ItWjyzSaNaAHR93yyDTKkoMgt"

# build download url
url="https://drive.google.com/uc?export=download&id=$download_id"




mkdir ../tmp

echo "downloading ..."
wget -O ../tmp/code.tar.gz $url

echo "extracting ..."
# Extract the contents of the zip file
mkdir ../tmp/extracted_files
tar xvfz  ../tmp/code.tar.gz -C ../tmp/extracted_files


#cache assets
echo "caching assets ..."
mkdir ../tmp/extracted_files/assets
cp -r ../code/assets ../tmp/extracted_files/assets

echo "deleting old code ..."
# Delete the old code
rm -rf ../code

echo "moving new code ..."
# Move the new code to the code folder
mv ../tmp/extracted_files ../code

echo "deleting tmp files ..."
# Delete the tmp files
rm -rf ../tmp

echo "change permissions ..."
chmod -R 777 ../

echo "done"
