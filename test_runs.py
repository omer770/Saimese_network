import os
import torch
import random
import torchvision 
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Tuple
import torchvision.transforms as T
from PIL import Image

idx_2_label = {0:'no',1:'rework',2:'solar',3:'tarp',4:'different'}
label_2_idx ={'no':0,'rework':1,'solar':2,'tarp':3,'different' : 4}
class_names = list(label_2_idx.keys())

def pred_and_plot_image(model: torch.nn.Module,
                        image_path1: str,
                        image_path2: str,
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device=device):


    # 2. Open image
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)

    # 3. Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = T.Compose([
          T.CenterCrop(512),
          T.Resize(image_size),
          T.ToTensor(),
        ])

    ### Predict on image ###

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
      # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
      transformed_image1 = image_transform(img1).unsqueeze(dim=0)
      transformed_image2 = image_transform(img2).unsqueeze(dim=0)

      # 7. Make a prediction on image with an extra dimension and send it to the target device
      target_image_pred = model(transformed_image1.to(device),transformed_image2.to(device))
    #print("target_image_pred: ",target_image_pred)
    # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 10. Plot image with predicted label and probability
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(transformed_image1.squeeze(dim=0).clamp(0, 1).permute(1, 2, 0).cpu().numpy())
    plt.axis(False)

    plt.subplot(1,2,2)
    plt.imshow(transformed_image2.squeeze(dim=0).clamp(0, 1).permute(1, 2, 0).cpu().numpy())
    plt.axis(False)

    plt.suptitle(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}");

def report_function(cls):
    if cls == 'no':return 'No significant changes on the roof.','No'
    elif cls == 'rework': return 'Rework done on the roof.','repair'
    elif cls == 'solar': return "Addition of solar panels","installation"
    elif cls == 'tarp': return 'There is a covering of tarpoline on the roof','modification'
    else: return 'Ambiguity','Different roof'

def str_2_lst(stng):
  stng = stng.replace("[","").replace("]","").replace("'","").replace(" ","").split(',')
  return stng

def pred_dict_results(dataframe:pandas.DataFrame):
  dict_id = {}
  for i in tqdm(range(len(dataframe))):
    building_id,buildings,years = dataframe.loc[i,:]
    buildings = str_2_lst(str(buildings))
    years = str_2_lst(str(years))
    dict_j = {}
    for j in range(len(buildings)):
      if j == 0:
        init_year = 'begining'
        curr_year = str(years[0])
        disc = 'No changes on the roof.'
        typ = 'initial'
        dict_j[j] = {
            'year': init_year+' - '+curr_year,
            "description": disc,
            "type": typ}
      else:
        init_year = str(years[j-1])
        curr_year = str(years[j])
        cls = predict(buildings[j-1],buildings[j])
        disc,typ = report_function(cls)
        dict_j[j] = {
            'year': init_year+' - '+curr_year,
            "description": disc,
            "type": typ}
    dict_id[building_id] = dict_j
  return dict_id
  
def predict(building1,building2,
            class_names: List[str] = class_names,
            root_dir = root_dir,
            image_size: Tuple[int, int] = (224, 224),
            transform: torchvision.transforms = None,
            device: torch.device=device):
  #print(building1,building2)
  # 2. Open image
  image_path1 = os.path.join(root_dir,building1)
  image_path2 = os.path.join(root_dir,building2)
  img1 = Image.open(image_path1)
  img2 = Image.open(image_path2)

  # 3. Create transformation for image (if one doesn't exist)
  if transform is not None:
      image_transform = transform
  else:
      image_transform = T.Compose([
                                  T.CenterCrop(512),
                                  T.Resize((224, 224)),
                                  T.ToTensor(),])

  ### Predict on image ###

  # 4. Make sure the model is on the target device
  model.to(device)

  # 5. Turn on model evaluation mode and inference mode
  model.eval()
  with torch.inference_mode():
    # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
    transformed_image1 = image_transform(img1).unsqueeze(dim=0)
    transformed_image2 = image_transform(img2).unsqueeze(dim=0)

    # 7. Make a prediction on image with an extra dimension and send it to the target device
    target_image_pred = model(transformed_image1.to(device),transformed_image2.to(device))
  #print("target_image_pred: ",target_image_pred)
  # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
  target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

  # 9. Convert prediction probabilities -> prediction labels
  target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
  cls = class_names[target_image_pred_label]
  return cls
