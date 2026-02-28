import torch
from torch import nn
from torchsummary import summary

    
class CNN(nn.Module):
    """
    CNN with 3 convolutions layers and on dense layer
    all layers are configurable, and the dimension of the dense layer is automatically calculated
    """
    def __init__(self, model_params = {
                 'conv_filters': (64, 128, 256), 
                 'kernel_size': (5, 3, 3), 
                 'pooling': (2, 2, 2), 
                 'dense_neurons': 128,
                 'dropout': (0.2, 0.2, 0.2, 0.2)
                 }):
  
        conv_filters = model_params['conv_filters']
        kernel_size = model_params['kernel_size']
        pooling = model_params['pooling']
        dense_neurons = model_params['dense_neurons']
        dropout = model_params['dropout']

        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, conv_filters[0], kernel_size=kernel_size[0]),
            nn.Dropout(dropout[0]),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=pooling[0])
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv_filters[0], conv_filters[1], kernel_size=kernel_size[1]),
            nn.Dropout(dropout[1]),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=pooling[1])
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(conv_filters[1], conv_filters[2], kernel_size=kernel_size[2]),
            nn.Dropout(dropout[2]),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=pooling[2])
        )

        conv_output_size = self._calculate_conv_output_size(300, kernel_size, pooling)

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_filters[2] * conv_output_size, dense_neurons),
            nn.Dropout(dropout[3]),
            nn.LeakyReLU()
        )

        self.output = nn.Sequential(
            nn.Linear(dense_neurons, 1),
            nn.Sigmoid()
        )

    def _calculate_conv_output_size(self, input_size, kernel_size, pooling):
        """Calculate output size after 3 conv layers and 1 pooling"""
        size = input_size
        for i in range(3):
            size = size - (kernel_size[i] - 1)
            size = size // pooling[i]
        return size*size

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dense(x)
        x = self.output(x)
        return x

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\nRésumé du modèle CNN:\n")
    cnn = CNN()
    cnn.to(device)
    summary(cnn, input_size=(1, 300, 300), device=device)
    print('---'*30)
    print(cnn)
