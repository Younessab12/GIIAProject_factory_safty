#!/bin/bash

# Download the zip file from Google Drive
absolute_url="" #$1

# extract download id
# ex https://drive.google.com/file/d/1MyDe5__ItWjyzSaNaAHR93yyDTKkoMgt/view?usp=drive_link
# to 1MyDe5__ItWjyzSaNaAHR93yyDTKkoMgt
download_id=$(echo $absolute_url | cut -d'/' -f6)
download_id="1wpmnz09zj1_0ZhgD_swQJrdltZorNVdK"

# build download url
url="https://drive.google.com/uc?export=download&id=$download_id"

mkdir ../tmp

echo "downloading ..."
wget -O ../tmp/assets.tar.gz $url

echo "extracting ..."
# Extract the contents of the zip file
mkdir ../tmp/extracted_files
tar xvfz  ../tmp/assets.tar.gz -C ../tmp/extracted_files

echo "deleting old assets ..."
# Delete the old code
rm -rf ../code/assets

echo "moving new assets ..."
# Move the new code to the code folder
mv ../tmp/extracted_files ../code/assets

echo "deleting tmp files ..."
# Delete the tmp files
rm -rf ../tmp

echo "change permissions ..."
chmod -R 777 ../

echo "done"

