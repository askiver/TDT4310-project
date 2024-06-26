import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error


def add_gaussian_noise(data, std=6):
    # Create noise with the same shape as the batch
    noise = torch.randn_like(data) * std
    return data + noise


class Trainer:

    def __init__(self, model, device, lr=0.0001, weight_decay=1e-4, binary_classification=True):
        self.model = model
        self.model = self.model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.binary_classification = binary_classification
        # Binary cross entropy loss for binary classification, MAE loss for regression
        self.criterion = torch.nn.BCEWithLogitsLoss() if binary_classification else torch.nn.L1Loss()

        # Loss history for plotting
        self.loss_train_history = []
        self.loss_test_history = []
        self.test_accuracy_history = []

    def train(self, train_loader, test_loader, epochs=50, save_models=False, plot_loss=False, add_noise=True,
              noise_std=6, spacy_model='bert'):

        # Train for epochs nr of times
        for i in range(epochs):
            self.model.train()

            train_loss = []
            for features, target in tqdm(train_loader, total=len(train_loader)):
                if add_noise:
                    features = add_gaussian_noise(features, noise_std)
                features, target = features.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, target)
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())

            test_loss = []
            self.model.eval()
            with torch.no_grad():
                for features, target in tqdm(test_loader, total=len(test_loader)):
                    features, target = features.to(self.device), target.to(self.device)
                    outputs = self.model(features)
                    loss = self.criterion(outputs, target)
                    test_loss.append(loss.item())

            test_accuracy = 0
            mae = 0
            self.model.eval()
            with torch.no_grad():
                for features, target in tqdm(test_loader, total=len(test_loader)):
                    features, target = features.to(self.device), target.to(self.device)
                    if self.binary_classification:
                        outputs = self.model.forward_accuracy(features)
                        test_accuracy += torch.sum(outputs == target).item()
                    else:
                        outputs = self.model.forward_score(features)
                        test_accuracy += outputs.eq(target).sum().item()
                        mae += mean_absolute_error(target.cpu().numpy(), outputs.cpu().numpy())

            # Save loss
            self.loss_train_history.append((sum(train_loss) / len(train_loss)))
            self.loss_test_history.append((sum(test_loss) / len(test_loss)))
            self.test_accuracy_history.append((test_accuracy / len(test_loader.dataset)))
            if not self.binary_classification:
                mae = mae / sum(1 for _ in test_loader)

            # Print loss
            tqdm.write(
                f"Epoch {i + 1}/{epochs} - Train loss: {self.loss_train_history[-1]:.4f}, test loss: {self.loss_test_history[-1]:.4f}, Test accuracy: {self.test_accuracy_history[-1]:.4f}, MAE: {mae:.4f}")

            # Save model
            if save_models:
                # Only save the model if the test loss is the lowest so far
                if self.loss_test_history[-1] == min(self.loss_test_history):
                    torch.save(self.model.state_dict(), f"models/SimpleFlatCNN/{spacy_model}.pt")

        if plot_loss:
            self.plot_loss()

    def plot_loss(self):
        # Creates plots showing loss and accuracy
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Plot loss
        axs[0].plot(self.loss_train_history, label="Train loss")
        axs[0].plot(self.loss_test_history, label="Test loss")
        axs[0].set_title("Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].legend()

        # Plot accuracy
        axs[1].plot(self.test_accuracy_history, label="Test accuracy")
        axs[1].set_title("Accuracy")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Accuracy")
        axs[1].legend()

        # Display the plot
        plt.tight_layout()
        plt.show()
