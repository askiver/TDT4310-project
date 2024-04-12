import json
import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def load_data(file_path):
    with h5py.File(file_path, 'r') as hdf:
        data = {key: hdf[key][()] for key in hdf.keys()}
    return data


def load_movie_reviews(spacy_model='sm', data_size=0.2, data_type='train', data_label='pos'):
    data = load_data(f'data/embeddings/{data_type}/{data_label}_spacy_model_{spacy_model}_embeddings.h5')

    data = data[data_label][:int(len(data[data_label]) * data_size)]

    if data_label == 'pos':
        labels = np.ones(len(data))
    else:
        labels = np.zeros(len(data))

    return data, labels


# Class for loading word embeddings from file
def create_tensor_dataset(spacy_model='sm', batch_size=32, test_size=0.2, train_size=1.0, flat_tensor=False):
    # Load data
    positive_embeddings_train, positive_labels_train = load_movie_reviews(spacy_model, data_size=train_size,
                                                                          data_type='train', data_label='pos')
    negative_embeddings_train, negative_labels_train = load_movie_reviews(spacy_model, data_size=train_size,
                                                                          data_type='train', data_label='neg')
    positive_embeddings_test, positive_labels_test = load_movie_reviews(spacy_model, data_size=test_size,
                                                                        data_type='test', data_label='pos')
    negative_embeddings_test, negative_labels_test = load_movie_reviews(spacy_model, data_size=test_size,
                                                                        data_type='test', data_label='neg')

    # Combine data
    embeddings_train = np.concatenate([positive_embeddings_train, negative_embeddings_train], axis=0)
    labels_train = np.concatenate([positive_labels_train, negative_labels_train], axis=0)

    embeddings_test = np.concatenate([positive_embeddings_test, negative_embeddings_test], axis=0)
    labels_test = np.concatenate([positive_labels_test, negative_labels_test], axis=0)

    # Convert to tensors
    embeddings_train = torch.tensor(embeddings_train).float()
    embeddings_test = torch.tensor(embeddings_test).float()
    labels_train = torch.tensor(labels_train).float()
    labels_test = torch.tensor(labels_test).float()

    if flat_tensor:
        embeddings_train = embeddings_train.transpose(1, 2)
        embeddings_test = embeddings_test.transpose(1, 2)
    else:
        # insert channel dimension
        embeddings_train = embeddings_train.unsqueeze(1)
        embeddings_test = embeddings_test.unsqueeze(1)

    # Create dataset
    dataset_train = TensorDataset(embeddings_train, labels_train)
    dataset_test = TensorDataset(embeddings_test, labels_test)

    # create dataloader
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    return dataloader_train, dataloader_test
