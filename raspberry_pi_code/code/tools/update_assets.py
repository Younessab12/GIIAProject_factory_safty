import requests
import shutil
import tarfile
import os

def update_assets():
  print("downloading ...")
  
  url="https://drive.usercontent.google.com/download?id=1wpmnz09zj1_0ZhgD_swQJrdltZorNVdK&export=download&authuser=0&confirm=t&uuid=d2120172-9960-4d47-86e1-cfa9b0be07dc&at=APZUnTVh_zMYsK2NbHABQ9KAs8_c:1705621470178"
  response = requests.get(url)
  if not os.path.exists("tmp"):
    os.makedirs("tmp")
  tar_file_path = "tmp/assets.tar.gz"

  print("saving ...")
  with open(tar_file_path, "wb") as tar_file:
    tar_file.write(response.content)

  print("deleting old assets ...")
  old_code_path = "assets"
  if os.path.exists(old_code_path):
    shutil.rmtree(old_code_path)

  if not os.path.exists("assets"):
    os.makedirs("assets")

  print("extracting ...")
  with tarfile.open(tar_file_path, 'r') as tar:
    tar.extractall("assets")

  print("deleting tmp files ...")
  shutil.rmtree("tmp")

  print("done")