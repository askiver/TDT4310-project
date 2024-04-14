import random
import numpy as np
import torch
from CNN import CNN, LargeCNN, FlatCNN, SimpleFlatCNN
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
BATCH_SIZE = 32
EPOCHS = 25
SPACY_MODEL = 'bert'
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-1
EMBEDDING_LENGTH = 200
VECTOR_DIMENSION = 768 if SPACY_MODEL == 'bert' else 300
TRAIN_SIZE = 1.0
TEST_SIZE = 0.2
ADD_NOISE = False
NOISE_STD = 0.4
BINARY_CLASSIFICATION = False


def train_model():
    torch.autograd.set_detect_anomaly(True)
    # Create model
    model = SimpleFlatCNN(binary_classification=BINARY_CLASSIFICATION, embedding_length=EMBEDDING_LENGTH, vector_dimension=VECTOR_DIMENSION)

    # set model to device
    model = model.to(DEVICE)

    # retrieve data
    dataloader_train, dataloader_test = create_tensor_dataset("train", SPACY_MODEL, BATCH_SIZE, train_size=TRAIN_SIZE, test_size=TEST_SIZE, flat_tensor=True, binary_classification=BINARY_CLASSIFICATION)

    # Create trainer
    trainer = Trainer(model, DEVICE, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, binary_classification=BINARY_CLASSIFICATION)

    # Train model
    trainer.train(dataloader_train, dataloader_test, epochs=EPOCHS, save_models=False, plot_loss=True,
                  add_noise=ADD_NOISE, noise_std=NOISE_STD)


def main():
    set_seed(SEED)
    train_model()


if __name__ == '__main__':
    main()
