import os
import spacy
from tqdm import tqdm
import h5py
from multiprocessing import Pool
from transformers import RobertaTokenizer, RobertaModel
import torch
from torch.utils.data import DataLoader, Dataset

class ReviewDataset(Dataset):
    def __init__(self, reviews, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.reviews = reviews
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        return self.tokenizer(review, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')


# File for creating the dataset
# Converts the reviews into word embeddings
def load_movie_reviews(folder):
    reviews = []
    for filename in os.listdir(folder):
        filepath = f'{folder}/{filename}'
        with open(filepath, 'r', encoding='utf-8') as file:
            review = file.read()
            reviews.append(review)
    return reviews


def remove_breaks(reviews):
    return [review.replace('<br /><br />', '') for review in reviews]


def create_embeddings_spacy(reviews, spacy_model):
    nlp = spacy.load(f'en_core_web_{spacy_model}')
    embeddings = []

    for doc in tqdm(nlp.pipe(reviews, batch_size=500), total=len(reviews)):
        embeddings.append([token.vector.tolist() for token in doc if not token.is_stop])
    return embeddings


def normalize_embedding_length(embeddings, chosen_length=100, vector_length=768):
    # Normalize the length of the embeddings
    normalized_embeddings = []
    for embedding in embeddings:
        if len(embedding) > chosen_length:
            normalized_embedding = embedding[:chosen_length]
        else:
            normalized_embedding = embedding + [[0] * vector_length] * (chosen_length - len(embedding))
        normalized_embeddings.append(normalized_embedding)
    return normalized_embeddings


def save_word_embeddings(data_type, data_label, word_embeddings, spacy_model):
    with h5py.File(f'data/embeddings/{data_type}/{data_label}_spacy_model_{spacy_model}_embeddings.h5', 'w') as hdf:
        hdf.create_dataset(data_label, data=word_embeddings)


def clear_file_contents(data_type, data_label, spacy_model):
    with h5py.File(f'data/embeddings/{data_type}/{data_label}_spacy_model_{spacy_model}_embeddings.h5', 'w') as hdf:
        pass


def create_embedding_bert(reviews):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Create the dataset and data loader
    dataset = ReviewDataset(reviews, tokenizer)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False)

    embeddings = []

    # Process batches
    for batch in tqdm(data_loader, desc="Generating Embeddings"):
        input_ids = batch['input_ids'].squeeze(1).to(device)
        attention_mask = batch['attention_mask'].squeeze(1).to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings.append(output.last_hidden_state.to('cpu'))

    # Concatenate all batch embeddings and move to CPU if needed
    embeddings = torch.cat(embeddings, dim=0).to('cpu')

    return embeddings


def create_dataset(data_type, data_label, spacy_model, embedding_length):
    reviews = load_movie_reviews(f'data/{data_type}/{data_label}')

    reviews = remove_breaks(reviews)

    if spacy_model == 'bert':
        embeddings = create_embedding_bert(reviews)
    else:
        embeddings = create_embeddings_spacy(reviews, spacy_model)

    embeddings = normalize_embedding_length(embeddings, embedding_length)

    save_word_embeddings(data_type, data_label, embeddings, spacy_model)


def main():
    spacy_model = 'bert'
    data_types = ['train', 'test']
    labels = ['pos', 'neg']
    embedding_length = 200
    vector_length = 768
    # clear files
    for data_type in data_types:
        for label in labels:
            #clear_file_contents(data_type, label, spacy_model)
            create_dataset(data_type, label, spacy_model=spacy_model, embedding_length=embedding_length)
            print(f'Finished creating {data_type} {label} dataset')


if __name__ == '__main__':
    main()
