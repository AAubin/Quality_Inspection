from datetime import datetime
from tqdm import tqdm
import json
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from cnn import CNN
from early_stopping import EarlyStopping

def train_single_epoch(model: nn.Module, data_loader: DataLoader, val_data_loader: DataLoader, loss_fn, optimizer, device, scheduler=None):
    epoch_loss = 0
    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    for input, target in progress_bar:
        input, target = input.to(device), target.to(device)
        # calculate loss
        prediction = model(input).squeeze()
        loss = loss_fn(prediction, target.float())

        # backpropagate error and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    #Evaluation on val_data_loader
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_val, y_val in tqdm(val_data_loader, desc="Validation", leave=False):
            X_val, y_val = X_val.to(device), y_val.to(device)
            y_hat = model(X_val).squeeze()
            loss = loss_fn(y_hat, y_val.float())
            val_loss += loss.item()
            progress_bar.set_postfix({'val loss': f'{loss.item():.4f}'})
    val_loss /= len(data_loader)
    if scheduler:
        scheduler.step(val_loss)
    train_loss = epoch_loss / len(data_loader)
    return train_loss, val_loss

def train(model, data_loader, val_data_loader, loss_fn, optimizer, device, epochs, scheduler=None, early_stopping=None):
    model.train()
    train_loss_hist, val_loss_hist = [], []
    for epoch in tqdm(range(epochs), desc='Epochs'):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, val_loss = train_single_epoch(model, data_loader, val_data_loader, loss_fn, optimizer, device, scheduler=scheduler)
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        print(f"\nTrain loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        if early_stopping:
            early_stopping(val_loss, model, optimizer, epoch)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        print('---'*30)
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Best model from epoch {checkpoint['epoch']+1} with val_loss: {checkpoint['val_loss']:.4f}")
    return {'train_loss': train_loss_hist, 'val_loss': val_loss_hist}

def create_train_val_dataloaders(data_dir, batch_size):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((300, 300)), transforms.ToTensor()])
    dataset_full = datasets.ImageFolder(root=data_dir+'train', transform=transform)

    indices = list(range(len(dataset_full)))
    labels = dataset_full.targets  
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    dataset_train = Subset(dataset_full, train_indices)
    dataset_val   = Subset(dataset_full, val_indices)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    return dataloader_train, dataloader_val


if __name__ == "__main__":
    # define hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 30
    LEARNING_RATE = 0.0001

    DATA_DIR = "./data/casting_data/"
    OUTPUT_DIR = "./results/trained_models/"
    TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M')
    print(TIMESTAMP)

    model_path = OUTPUT_DIR + f"trainedCNN_{TIMESTAMP}.pth"
    history_path = OUTPUT_DIR + f"history_{TIMESTAMP}.pkl"
    model_params_path = OUTPUT_DIR + f"model_params_{TIMESTAMP}.json"

    # Detect if an NVIDIA GPU is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")

    # instantiating our dataset object and create data loader for train and val
    dataloader_train, dataloader_val = create_train_val_dataloaders(DATA_DIR, BATCH_SIZE)
    
    # construct model and assign it to device
    model_params = {
        'conv_filters': (128, 256, 512),
        'kernel_size': (5, 3, 3),
        'pooling': (2, 2, 2),
        'dense_neurons': 256,
        'dropout': (0.2, 0.2, 0.2, 0.2)
    }
    with open(model_params_path, "w") as f:
        json.dump(model_params, f)
    model = CNN(model_params).to(device)
    print(model)

    # initialise loss funtion, optimizer, scheduler and early_stopping
    loss_fn = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                            mode='min',
                                            factor=0.1,
                                            patience=3)
    early_stopping = EarlyStopping(patience=7,
                            verbose=True,
                            delta=0.0001,
                                path='best_model.pt')
    # train model
    history = train(model, dataloader_train, dataloader_val, loss_fn, optimizer, device, EPOCHS, scheduler, early_stopping)
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved at {model_path}")
    pickle.dump(history, open(history_path, 'wb'))
    print(f"History data saved at {history_path}")

