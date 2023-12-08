import requests
import shutil
import zipfile

# Download the zip file from Google Drive
url = input("Enter the URL of the file to download: ex: https://drive.google.com/uc?id=YOUR_FILE_ID")

print("downloading ...")
response = requests.get(url)
zip_file_path = "../tmp/compressed.zip"

print("saving ...")
with open(zip_file_path, "wb") as zip_file:
  zip_file.write(response.content)

print("extracting ...")
# Extract the contents of the zip file
extracted_files_path = "path/to/save/extracted_files"
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
  zip_ref.extractall(extracted_files_path)

print("deleting old code ...")
# Delete the old code
old_code_path = "../code"
shutil.rmtree(old_code_path)

print("moving new code ...")
# Move the new code to the code folder
shutil.move(extracted_files_path, old_code_path)

print("deleting tmp files ...")
# Delete the tmp files
shutil.rmtree("../tmp")

print("done")
