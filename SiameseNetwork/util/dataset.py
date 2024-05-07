import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch.utils.data as Data
from typing  import  Tuple, Dict, List

idx_2_label = {0:'no',1:'rework',2:'solar',3:'tarp',4:'different'}
label_2_idx ={'no':0,'rework':1,'solar':2,'tarp':3,'different' : 4}
class_names = list(label_2_idx.keys())

def split_train_test_df(dataframe:pd.DataFrame,test_size:float = 0.2):
    df_train = dataframe.sample(frac=0.8)
    df_test = dataframe.drop(df_train.index)
    return df_train,df_test
    
class RoofDataset(Data.Dataset):
    def __init__(self, dataframe,root_dir,
                 label_2_idx:dict = label_2_idx,
                 idx_2_label:dict = idx_2_label,
                 transform=None,device='cpu'):
        self.df = dataframe
        self.transform = transform
        self.root_dir = root_dir
        self.device = device
        self.label_2_idx = label_2_idx
        self.idx_2_label = idx_2_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path1 = os.path.join(self.root_dir, self.df.loc[index,'Image1_path'])
        image1 = Image.open(img_path1)
        img_path2 = os.path.join(self.root_dir, self.df.loc[index,'Image2_path'])
        image2 = Image.open(img_path2)
        changes = self.df.loc[index,'changes']
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        ecd_labels = self.encode_label_onehot(changes)
        return image1, image2, ecd_labels

    def encode_label_onehot(self,label):
      num_classes = len(self.label_2_idx)
      onehot_vector = torch.zeros(num_classes).cpu()
      onehot_vector[self.label_2_idx[label]] = 1
      return onehot_vector

    def decode_label_onehot(self, onehot_labels):
      idx = onehot_labels.argmax().cpu().item()
      decoded_label = self.idx_2_label[idx]
      return decoded_label

def jsonify_str(strng:str ,path:str = None)->str:
  import json
  import os
  if path:
    current_dir = path
  else:
    current_dir = os.getcwd()
  remove = ["```json\n{\n  ","}\n","  ","```",'"','\n']
  for item in remove: strng = strng.replace(item,"")
  lst1 = strng.split(",")
  dictn = {lst.split(':')[0].strip():lst.split(':')[1].strip() for lst in lst1}
  # Convert and write JSON object to file
  path_2_json = os.path.join(current_dir,"output.json")
  with open(path_2_json, "w") as outfile:
      json.dump(dictn, outfile)
  return path_2_json
