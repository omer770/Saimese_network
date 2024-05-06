import os
import torch
import random
import torchvision 
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Tuple
import torchvision
from PIL import Image

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
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
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
