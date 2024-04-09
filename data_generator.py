import os
import spacy
from tqdm import tqdm
import h5py
from multiprocessing import Pool


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


def create_embeddings_chunk(reviews_chunk, spacy_model):
    nlp = spacy.load(f'en_core_web_{spacy_model}')
    embeddings_chunk = []

    for doc in tqdm(nlp.pipe(reviews_chunk, batch_size=50), total=len(reviews_chunk)):
        if spacy_model == 'trf':
            transformer_output = doc._.trf_data.last_hidden_layer_state
            embeddings_chunk.append([word_embedding.tolist() for word_embedding in transformer_output])
        else:
            embeddings_chunk.append([token.vector.tolist() for token in doc if not token.is_stop])
    return embeddings_chunk


def create_embeddings(reviews, spacy_model):
    num_processes = 1 # os.cpu_count()
    # Split the reviews into chunks, one for each process
    chunk_size = len(reviews) // num_processes
    reviews_chunks = [reviews[i:i + chunk_size] for i in range(0, len(reviews), chunk_size)]

    # Create a pool of worker processes
    with Pool(processes=num_processes) as pool:
        # Map (apply) the create_embeddings_chunk function to each chunk
        results = pool.starmap(create_embeddings_chunk, [(chunk, spacy_model) for chunk in reviews_chunks])

    # Flatten the list of lists
    word_embeddings = [embedding for chunk in results for embedding in chunk]

    return word_embeddings


def normalize_embedding_length(embeddings, chosen_length=100):
    # Normalize the length of the embeddings
    normalized_embeddings = []
    for embedding in embeddings:
        if len(embedding) > chosen_length:
            normalized_embedding = embedding[:chosen_length]
        else:
            normalized_embedding = embedding + [[0] * 300] * (chosen_length - len(embedding))
        normalized_embeddings.append(normalized_embedding)
    return normalized_embeddings


def save_word_embeddings(data_type, data_label, word_embeddings, spacy_model):
    with h5py.File(f'data/embeddings/{data_type}/{data_label}_spacy_model_{spacy_model}_embeddings.h5', 'w') as hdf:
            hdf.create_dataset(data_label, data=word_embeddings)


def clear_file_contents(data_type, data_label, spacy_model):
    with h5py.File(f'data/embeddings/{data_type}/{data_label}_spacy_model_{spacy_model}_embeddings.h5', 'w') as hdf:
        pass


def create_dataset(data_type, data_label, spacy_model, embedding_length):
    reviews = load_movie_reviews(f'data/{data_type}/{data_label}')[:50]

    reviews = remove_breaks(reviews)

    #embeddings = create_embeddings(reviews, spacy_model)

    embeddings = create_embeddings_chunk(reviews, spacy_model)

    embeddings = normalize_embedding_length(embeddings, embedding_length)

    save_word_embeddings(data_type, data_label, embeddings, spacy_model)


def main():
    spacy_model = 'trf'
    data_types = ['train', 'test']
    labels = ['pos', 'neg']
    embedding_length = 200
    # clear files
    for data_type in data_types:
        for label in labels:
            #clear_file_contents(data_type, label, spacy_model)
            create_dataset(data_type, label, spacy_model=spacy_model, embedding_length=embedding_length)
            print(f'Finished creating {data_type} {label} dataset')


if __name__ == '__main__':
    main()
