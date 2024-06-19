import os


from torch.utils.data import Dataset
import pandas as pd
import torch

class WSI_dataset(Dataset):
    def __init__(self,wsi_list,features_path):
        self.wsi_list = wsi_list
        self.features_path = features_path
        self.label_to_int = {'normal_tissue': 0, 'tumor_tissue': 1}
    def __len__(self):
        return len(self.wsi_list)

    def __getitem__(self, idx):
        name = self.wsi_list[idx]['slide_id']
        label = torch.tensor(self.label_to_int[self.wsi_list[idx]['label']], dtype=torch.long)
        features = torch.load(self.features_path+name+'_features.pt')
        return (features,label)

%matplotlib inline
import matplotlib.pyplot as plt

def plot_loss(train_loss, val_loss, epochs):
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_loss, label='Train Loss', color='blue')

    plt.plot(epochs, val_loss, label='Validation Loss', color='red')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
