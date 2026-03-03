import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cnn import CNN
from simple_train import create_train_val_dataloaders
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

def metrics_evaluation(model, data_loader, criterion, device, seuil=0.5):
    # Passer le modèle en évaluation
    model.eval()
    # Calculer la loss totale
    loss_val_total = 0
    # Stocker les prédictions et les vraies valeurs.
    predictions, true_vals = [], []
    for batch in data_loader:
        X_batch, y_batch = batch
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        with torch.no_grad():
            # Prédiction du modèle pour un batch donné
            y_pred = model(X_batch).squeeze()
        # Calcul de la fonction de perte pour l'utiliser comme une métrique
        loss = criterion(y_pred, y_batch.float())
        # Cumuler la fonction de perte de tous les lots de données.
        loss_val_total += loss.item()
        #Convertir les probabilités en classes prédictes
        binary_preds = (y_pred >= seuil).float()
        # Enregistrer les prédictions pour les utiliser plus tard
        predictions.extend(binary_preds.cpu().numpy())
        # Enregistrer les vraies valeurs pour les utiliser plus tard
        true_vals.extend(y_batch.cpu().numpy())

    # Loss du jeu de données val
    loss_val_avg = loss_val_total / len(data_loader)
    # Ensemble des prédictions du jeu de données
    predictions = np.array(predictions)
    # Ensemble des vraies valeurs du jeu de données
    true_vals = np.array(true_vals)
    precision, recall, f1_score, _ = precision_recall_fscore_support(true_vals, predictions, zero_division=0)
    metrics = {
        "loss":loss_val_avg, 
        "accuracy":accuracy_score(true_vals, predictions),
        "precision_class_0":precision[0],
        "precision_class_1":precision[1],
        "recall_class_0":recall[0],
        "recall_class_1":recall[1],
        "f1_score_class_0":f1_score[0],
        "f1_score_class_1":f1_score[1]
    }
    cm = confusion_matrix(true_vals, predictions)
    return {"metrics": metrics, "confusion_matrix": cm}

def create_test_dataloader(data_dir, batch_size):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((300, 300)), transforms.ToTensor()])
    dataset_test = datasets.ImageFolder(root=data_dir+'test', transform=transform)

    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    return dataloader_test

def save_metrics(metrics, dataset, timestamp, res_dir):
    metrics_dir = res_dir + f"res_{timestamp}/"
    # Créer le répertoire s'il n'existe pas
    Path(metrics_dir).mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir + f"metrics_{dataset}.json"
    print(f"Saving {dataset} metrics to {metrics_path}")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__=='__main__':
    BATCH_SIZE = 32
    TIMESTAMP = "20260227_1909"
    DATA_DIR = "./data/casting_data/"
    RESULTS_DIR = "./results/evaluation_results/"

    model_path = f"./results/trained_models/trainedCNN_{TIMESTAMP}.pth"
    model_params_path = f"./results/trained_models/model_params_{TIMESTAMP}.json"
    # Detect if an NVIDIA GPU is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")

    # Load trained model
    print(f"Loading model from {model_path}")
    with open(model_params_path, 'r') as f:
        saved_params = json.load(f)
    model = CNN(saved_params).to(device)
    saved_weights = torch.load(model_path)
    model.load_state_dict(saved_weights)

    # Create datasets
    dataloader_train, dataloader_val = create_train_val_dataloaders(DATA_DIR, BATCH_SIZE)
    dataloader_test = create_test_dataloader(DATA_DIR, BATCH_SIZE)

    loss_fn = torch.nn.BCELoss()

    print('\nEvaluate model on train set')
    train_res = metrics_evaluation(model, dataloader_train, loss_fn, device, seuil=0.5)
    print(f"Train metrics: {train_res['metrics']}")
    print(f"Confusion Matrix:\n{train_res['confusion_matrix']}")
    save_metrics(train_res['metrics'], "train", TIMESTAMP, RESULTS_DIR)
    print('---'*30)

    print('\nEvaluate model on val set')
    val_res = metrics_evaluation(model, dataloader_val, loss_fn, device, seuil=0.5)
    print(f"Val metrics: {val_res['metrics']}")
    print(f"Confusion Matrix:\n{val_res['confusion_matrix']}")
    save_metrics(val_res['metrics'], "val", TIMESTAMP, RESULTS_DIR)
    print('---'*30)

    print('\nEvaluate model on test set')
    test_res = metrics_evaluation(model, dataloader_test, loss_fn, device, seuil=0.5)
    print(f"Test metrics: {test_res['metrics']}")
    print(f"Confusion Matrix:\n{test_res['confusion_matrix']}")
    save_metrics(test_res['metrics'], "test", TIMESTAMP, RESULTS_DIR)
    print('---'*30)
