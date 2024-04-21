import random
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, classification_report
from CNN import SimpleFlatCNN
from data_loader import create_tensor_dataset
from trainer import Trainer


def set_seed(seed=0):
    # Include these lines to make the script reproducible
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# global params
SEED = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
EPOCHS = 10
SPACY_MODEL = 'lg'
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-1
EMBEDDING_LENGTH = 200
VECTOR_DIMENSION = 768 if SPACY_MODEL == 'bert' else 300
TRAIN_SIZE = 1.0
TEST_SIZE = 0.2
ADD_NOISE = True
NOISE_STD = 0.25
BINARY_CLASSIFICATION = True


def train_model():
    # retrieve data
    dataloader_train = create_tensor_dataset("train", SPACY_MODEL, BATCH_SIZE, data_size=TRAIN_SIZE, flat_tensor=True,
                                             binary_classification=BINARY_CLASSIFICATION)
    dataloader_test = create_tensor_dataset("test", SPACY_MODEL, BATCH_SIZE, data_size=TEST_SIZE, flat_tensor=True,
                                            binary_classification=BINARY_CLASSIFICATION)

    # Create model
    model = SimpleFlatCNN(binary_classification=BINARY_CLASSIFICATION, embedding_length=EMBEDDING_LENGTH,
                          vector_dimension=VECTOR_DIMENSION)

    # set model to device
    model = model.to(DEVICE)

    # Create trainer
    trainer = Trainer(model, DEVICE, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                      binary_classification=BINARY_CLASSIFICATION)

    # Train model
    trainer.train(dataloader_train, dataloader_test, epochs=EPOCHS, save_models=True, plot_loss=True,
                  add_noise=ADD_NOISE, noise_std=NOISE_STD, spacy_model=SPACY_MODEL)


def test_model():
    # Create model
    model = SimpleFlatCNN(binary_classification=BINARY_CLASSIFICATION, embedding_length=EMBEDDING_LENGTH,
                          vector_dimension=VECTOR_DIMENSION)

    # Load saved model
    model.load_state_dict(torch.load(f'models/SimpleFlatCNN/{SPACY_MODEL}.pt'))

    # set model to device
    model = model.to(DEVICE)

    # Load test set
    dataloader_test = create_tensor_dataset("test", SPACY_MODEL, BATCH_SIZE, data_size=TEST_SIZE, flat_tensor=True,
                                            binary_classification=BINARY_CLASSIFICATION)

    MAE = 0
    test_accuracy = 0
    all_outputs = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for features, target in dataloader_test:
            features, target = features.to(DEVICE), target.to(DEVICE)
            if BINARY_CLASSIFICATION:
                outputs = model.forward_accuracy(features)
                all_outputs.extend(outputs.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                test_accuracy += torch.sum(outputs == target).item()
            else:
                outputs = model.forward_score(features)
                test_accuracy += outputs.eq(target).sum().item()
                MAE += mean_absolute_error(target.cpu().numpy(), outputs.cpu().numpy())

    print(f"Test accuracy: {test_accuracy / len(dataloader_test.dataset):.4f}")

    if BINARY_CLASSIFICATION:
        print(f"classification report: {classification_report(all_targets, all_outputs, digits=4)}")
    else:
        print(f"MAE: {MAE / len(dataloader_test):.4f}")


def main():
    set_seed(SEED)
    train_model()
    test_model()


if __name__ == '__main__':
    main()
