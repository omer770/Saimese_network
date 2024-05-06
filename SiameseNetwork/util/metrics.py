import torch
from typing import List
from pprint import pprint
from torchmetrics import ConfusionMatrix
from torch.nn import functional as F

idx_2_label = {0:'no',1:'rework',2:'solar',3:'tarp',4:'different'}
label_2_idx ={'no':0,'rework':1,'solar':2,'tarp':3,'different' : 4}
class_names = list(label_2_idx.keys())

def contrastive_loss(similarity, label):
    margin = 1.0  # Hyperparameter to adjust
    loss = torch.mean((1 - label) * similarity ** 2 
                      + label * F.relu(margin - similarity) ** 2)
    return loss
  
def calculate_ConfusionMatrices(y_pred_tensors:List,targets_tensors:List,
                                idx_2_label:dict =idx_2_label,
                                class_names:List =class_names):
  confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
  confmat_tensor = confmat(preds=y_pred_tensor,target=targets_tensors)

  print("Classes: ",end=' ')
  pprint(class_names)
  print("Confusion Matrix: ")
  pprint(conf_mat)
  print('-'*75) 

