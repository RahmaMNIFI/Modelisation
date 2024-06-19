import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.model_clam import CLAM_SB
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from util import WSI_dataset


def evaluate_model(model, data_loader, loss_fn, device):
    model.eval()
    total_correct = 0
    TP = 0    
    TN = 0   
    FP = 0   
    FN = 0  
    test_loss = 0.0
    all_preds_prob = []
    all_labels = []

    with torch.no_grad():
        for features, label in tqdm(data_loader):
            features = features.to(device)
            label = label.to(device)
            
            logits, Y_prob, _, _, _ = model(features, label=label, instance_eval=False)
            loss = loss_fn(logits, label)
            test_loss += loss.item()
            
            Y_prob = Y_prob.cpu().numpy()
            prediction = Y_prob.argmax(axis=1)
            
            all_preds_prob.extend(Y_prob[:, 1])
            all_labels.extend(label.cpu().numpy())
            
            total_correct += (prediction == label.cpu().numpy()).sum()
            TP += ((prediction == 1) & (label.cpu().numpy() == 1)).sum()
            TN += ((prediction == 0) & (label.cpu().numpy() == 0)).sum()
            FP += ((prediction == 1) & (label.cpu().numpy() == 0)).sum()
            FN += ((prediction == 0) & (label.cpu().numpy() == 1)).sum()

    precision = TP / (TP + FP) 
    recall = TP / (TP + FN) 
    f1 = 2 * (precision * recall) / (precision + recall) 
    accuracy = total_correct / len(data_loader.dataset)
    test_loss /= len(data_loader)

    return accuracy, precision, recall, f1, test_loss

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate CLAM model on test data')
    parser.add_argument('--wsi_labels_path', type=str, default='/content/drive/MyDrive/data/label_test1.csv',
                        help='Path to CSV file containing WSIs and labels')
    parser.add_argument('--features_path', type=str, default='/content/drive/MyDrive/features/test/',
                        help='Path to directory containing extracted features')
    parser.add_argument('--labels_path', type=str, default='/content/drive/MyDrive/data/reference.csv',
                        help='Path to CSV file containing reference labels')
    parser.add_argument('--model_state_dict_path', type=str, default='/content/drive/MyDrive/CLAM/weights_25epochs.pt',
                        help='Path to model state dictionary file')
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()

    wsi_df = pd.read_csv(args.wsi_labels_path)
    label_to_int = {'normal_tissue': 0, 'tumor_tissue': 1}
    test_list = wsi_df.to_dict('records')
    test_dataset = WSI_Dataset(test_list, args.features_path, label_to_int)

    batch_size = 1
    shuffle = True
    num_workers = 4
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = {"dropout": True, 'n_classes': 2, "size_arg": 'small'}
    model = CLAM_SB(**model_dict).to(device)

    ckpt = torch.load(args.model_state_dict_path)
    ckpt_clean = {key.replace('.module', ''): ckpt[key] for key in ckpt.keys() if 'instance_loss_fn' not in key}
    model.load_state_dict(ckpt_clean, strict=True)

    loss_fn = nn.CrossEntropyLoss()
    accuracy, precision, recall, f1, test_loss = evaluate_model(model, test_loader, loss_fn, device)

 
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
