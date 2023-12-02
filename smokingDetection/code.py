import os
import cv2
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np

class image_clearing:
    def __init__(self,path) -> None:
        self.path = path

    def delete_unlabled_images(self):
        labels = []
        for file in os.listdir(self.path):
            if file.endswith(".xml"):
                labels.append(file.split(".")[0])
        for file in os.listdir(self.path):
            if file.endswith(".jpg") and file.split(".")[0] not in labels:
                os.remove(os.path.join(self.path,file))
                print("Deleted: ", file)
        print("Done")

        
    def convert_to_jpg(self):
        for file in os.listdir(self.path):
            if file.endswith(".jpeg"):
                img = cv2.imread(self.path+str(file))
                cv2.imwrite(self.path+str(file[:-5])+"-converted.jpg", img)
                os.remove(self.path+file)
        print("Done")

    def rename(self):
        count=1
        for file in os.listdir(self.path):
            os.rename(self.path+file, self.path+"image"+str(count)+"."+file.split(".")[-1])
            count+=1
        print("Done")


    def change_xml_from_jpeg_to_jpg(self):
        xml_list = []
        for xml_file in glob.glob(self.path + '/*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            filename=root.find('filename').text
            if filename.split('.')[-1]=='jpeg':
                root.find('filename').text=filename.split('.')[0]+'.jpg'
                tree.write(xml_file)
                
        print("Done")


        