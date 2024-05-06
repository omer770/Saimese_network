import numpy as np
import torch
import random
from typing import List,Dict
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

def plot_test_results(test_labels,pred_classes,test_samples1,test_samples2,test_data_custom):
  # Plot predictions
  plt.figure(figsize=(15,10))
  plt.suptitle(f'Test results: ')
  nrows = 3
  ncols = 6
  for i,(sample0,sample1) in enumerate(zip(test_samples1,test_samples2)):
    # Create a subplot
    j = 2*i+1
    plt.subplot(nrows, ncols, j)
    targ_image_adjust1 = sample0.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    plt.imshow(targ_image_adjust1)
    pred_label = test_data_custom.decode_label_onehot(pred_classes[i])

    # Get the truth label (in text form, e.g. "T-shirt")
    truth_label = test_data_custom.decode_label_onehot(test_labels[i])

    # Create the title text of the plot
    title_text = f"Pred: {pred_label}"

    # Check for equality and change title colour accordingly
    clr ='g' if pred_label == truth_label else 'r'

    plt.title(title_text, fontsize=10, c= clr,loc='right') # green text if correct
    plt.axis(False)

    plt.subplot(nrows, ncols, j+1)
    targ_image_adjust2 = sample1.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    # Plot the target image
    plt.imshow(targ_image_adjust2)
    title_text = f"Truth: {truth_label}"
    plt.title(title_text, fontsize=10, c= clr,loc='left')
    # Find the prediction label (in text form, e.g. "Sandal")
    plt.axis(False)

  plt.savefig("results/test_results.png");


def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();
