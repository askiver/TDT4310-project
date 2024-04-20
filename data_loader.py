import json
import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def load_data(file_path, data_size):
    with h5py.File(file_path, 'r') as hdf:
        data_size = int(data_size * len(hdf['embeddings']))
        embeddings = hdf['embeddings'][:data_size]
        scores = hdf['scores'][:data_size]

    return embeddings, scores


def load_movie_reviews(binary_classification, spacy_model='sm', data_size=0.2, data_type='train', data_label='pos'):
    data, scores = load_data(f'data/embeddings/{data_type}/{data_label}_spacy_model_{spacy_model}_embeddings.h5',
                             data_size)

    if binary_classification:
        if data_label == 'pos':
            labels = np.ones(len(data), dtype=bool)
        else:
            labels = np.zeros(len(data), dtype=bool)
    else:
        labels = scores

    return data, labels


# Class for loading word embeddings from file
def create_tensor_dataset(data_type, spacy_model='sm', batch_size=32, data_size=0.2, flat_tensor=False,
                          binary_classification=True):
    data_length = int(25000 * data_size)
    vector_dimension = 768 if spacy_model == 'bert' else 300

    all_data = np.empty((data_length, 200, vector_dimension), dtype=np.float32)
    all_labels = np.empty(data_length, dtype=np.float32)
    # Load data
    all_data[:data_length // 2], all_labels[:data_length // 2] = load_movie_reviews(binary_classification, spacy_model,
                                                                                    data_size=data_size,
                                                                                    data_type=data_type,
                                                                                    data_label='pos')

    all_data[data_length // 2:], all_labels[data_length // 2:] = load_movie_reviews(binary_classification, spacy_model,
                                                                                    data_size=data_size,
                                                                                    data_type=data_type,
                                                                                    data_label='neg')

    # Convert to tensors
    embeddings = torch.from_numpy(all_data)
    labels = torch.from_numpy(all_labels)

    if flat_tensor:
        # Transpose tensor
        embeddings = embeddings.transpose(1, 2)

    else:
        # insert channel dimension
        embeddings = embeddings.unsqueeze(1)

    # Create dataset
    dataset = TensorDataset(embeddings, labels)

    # create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    return dataloader
