import os

import numpy as np
import spacy
from tqdm import tqdm
import h5py
from transformers import RobertaTokenizer, RobertaModel
import torch
from torch.utils.data import DataLoader, Dataset
from utils import load_combined_reviews


class ReviewDataset(Dataset):
    def __init__(self, reviews, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.reviews = reviews
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        return self.tokenizer(review, add_special_tokens=True, padding='max_length', truncation=True,
                              max_length=self.max_length, return_tensors='pt')


def create_embeddings_spacy(reviews, spacy_model, vector_length=300, embedding_length=200):
    nlp = spacy.load(f'en_core_web_{spacy_model}')
    embeddings = np.zeros((len(reviews), embedding_length, vector_length), dtype=np.float32)

    for ind, doc in tqdm(enumerate(nlp.pipe(reviews, batch_size=500)), total=len(reviews)):
        word_vectors = [token.vector for token in doc if not token.is_stop and token.has_vector][:embedding_length]
        vector_count = min(len(word_vectors), embedding_length)
        embeddings[ind, :, :vector_count] = word_vectors[:vector_count]
    return embeddings


def save_word_embeddings(data_type, data_label, word_embeddings, scores, spacy_model):
    with h5py.File(f'data/embeddings/{data_type}/{data_label}_spacy_model_{spacy_model}_embeddings.h5', 'w') as hdf:
        hdf.create_dataset('embeddings', data=word_embeddings, dtype=np.float32)
        hdf.create_dataset('scores', data=scores, dtype=np.int8)


def create_embedding_bert(reviews, model, tokenizer, vector_length=768, embedding_length=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Create the dataset and data loader
    dataset = ReviewDataset(reviews, tokenizer)
    data_loader = DataLoader(dataset, batch_size=180, shuffle=False, num_workers=8)

    embeddings = []

    # Process batches
    for batch in tqdm(data_loader, desc="Generating Embeddings"):
        input_ids = batch['input_ids'].squeeze(1).to(device)
        attention_mask = batch['attention_mask'].squeeze(1).to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_embeddings = output.last_hidden_state[:, :embedding_length]
            if batch_embeddings.shape[1] < embedding_length:
                padding = torch.zeros(
                    (batch_embeddings.shape[0], embedding_length - batch_embeddings.shape[1], batch_embeddings.shape[2]),
                    device=device)
                batch_embeddings = torch.cat([batch_embeddings, padding], dim=1)
            embeddings.append(batch_embeddings.to('cpu'))

    # Concatenate all batch embeddings and move to CPU if needed
    embeddings = torch.cat(embeddings, dim=0)

    return embeddings


def create_dataset(bert_model, bert_tokenizer, data_type, data_label, spacy_model, embedding_length, vector_length):
    reviews, scores = load_combined_reviews(data_type, data_label)

    # Limit amount of reviews for testing
    reviews = reviews
    scores = scores

    if spacy_model == 'bert':
        embeddings = create_embedding_bert(reviews, bert_model, bert_tokenizer)
    else:
        embeddings = create_embeddings_spacy(reviews, spacy_model)

    save_word_embeddings(data_type, data_label, embeddings, scores, spacy_model)


def main():
    spacy_model = 'bert'
    data_types = ['train', 'test']
    labels = ['pos', 'neg']
    embedding_length = 200
    vector_length = 768 if spacy_model == 'bert' else 300

    tokenizer = None
    model = None

    if spacy_model == 'bert':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base')
    for data_type in data_types:
        for label in labels:
            create_dataset(model, tokenizer, data_type, label, spacy_model=spacy_model,
                           embedding_length=embedding_length,
                           vector_length=vector_length)
            print(f'Finished creating {data_type} {label} dataset')


if __name__ == '__main__':
    main()
