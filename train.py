import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.model_clam import CLAM_SB
from topk.svm import SmoothTop1SVM
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from util import plot_loss,WSI_dataset
# Définition des arguments
parser = argparse.ArgumentParser(description='Train and evaluate CLAM model on Camelyon17 dataset')
parser.add_argument('--wsi_labels_path', type=str, default='/content/drive/MyDrive/data/label_test1.csv', help='Path to the CSV file containing WSI labels')
parser.add_argument('--features_path', type=str, default='/content/drive/MyDrive/features/test/', help='Path to the directory containing WSI features')
parser.add_argument('--model_state_dict_path', type=str, default='/content/drive/MyDrive/CLAM/weights_25epochs.pt', help='Path to the model state dictionary')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for training')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
parser.add_argument('--bag_weight', type=float, default=0.7, help='Weight for bag loss')
args = parser.parse_args()

wsi_df = pd.read_csv(args.wsi_labels_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_list = []
val_list = []

for _, row in wsi_df.iterrows():
    role = row['role']
    if role == 'train':
        train_list.append(row)
    elif role == 'val':
        val_list.append(row)

train_dataset = WSI_dataset(train_list, args.features_path)
val_dataset = WSI_dataset(val_list, args.features_path)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)


model_dict = {"dropout": True, 'n_classes': 2, "size_arg": 'small', 'k_sample': 8}
model = CLAM_SB(**model_dict).to(device)

ckpt = torch.load(args.model_state_dict_path)
ckpt_clean = {key.replace('.module', ''): ckpt[key] for key in ckpt.keys() if 'instance_loss_fn' not in key}
model.load_state_dict(ckpt_clean, strict=True)

loss_fn = nn.CrossEntropyLoss()
instance_loss_fn = SmoothTop1SVM(n_classes=2).cuda()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0002, weight_decay=1e-5)

model.train()
train_loss_list = []
val_loss_list = []
best_f1 = 0
best_loss = float('inf')
early_stopping_counter = 0

for epoch in range(args.num_epochs):
    train_loss = 0.0
    val_loss = 0.0

    
    model.train()
    for features, label in tqdm(train_loader):
        features = features.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        logits, _, _, _, instance_dict = model(features, label=label, instance_eval=True)
        loss = loss_fn(logits, label)
        instance_loss = instance_dict['instance_loss']
        total_loss = args.bag_weight * loss + (1 - args.bag_weight) * instance_loss

        train_loss += loss.item()
        total_loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    ACC = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    all_preds_prob = []
    all_labels = []

    for features, label in tqdm(val_loader):
        features = features.to(device)
        label = label.to(device)

        with torch.no_grad():
            logits, Y_prob, _, _, instance_dict = model(features, label=label, instance_eval=False)
            loss = loss_fn(logits, label)
            val_loss += loss.item()
            all_preds_prob.append(Y_prob[0][1].cpu())
            all_labels.append(label.cpu())
            pred = 1 if Y_prob[0][1] > 0.49 else 0

            if pred == label.item():
                total_acc += 1
                if pred == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if pred == 1:
                    FP += 1
                else:
                    FN += 1

    precision = TP / (TP + FP) if TP
    recall = TP / (TP + FN) if TP 
    f1 = 2 * precision * recall / (precision + recall)
    ACC /= len(val_loader)
    val_loss /= len(val_loader)
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)

    
    
    print(f'accuracy: {ACC:.2f}, precision: {precision:.2f}, recall: {recall:.2f}, f1: {f1:.2f}')


plot_loss(train_loss_list, val_loss_list, range(1, epoch + 2))
