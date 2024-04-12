import random
import numpy as np
import torch
from CNN import CNN, LargeCNN, FlatCNN
from data_loader import create_tensor_dataset
from trainer import Trainer


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# global params
SEED = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-1
EMBEDDING_LENGTH = 200
SPACY_MODEL = 'lg'


def train_model():
    # Create model
    model = CNN(embedding_length=EMBEDDING_LENGTH)

    # set model to device
    model = model.to(DEVICE)

    # retrieve data
    dataloader_train, dataloader_test = create_tensor_dataset(SPACY_MODEL, BATCH_SIZE, flat_tensor=False)

    # Create trainer
    trainer = Trainer(model, DEVICE, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Train model
    trainer.train(dataloader_train, dataloader_test, epochs=EPOCHS, save_models=False, plot_loss=True, add_noise=False)


def main():
    set_seed(SEED)
    train_model()


if __name__ == '__main__':
    main()
