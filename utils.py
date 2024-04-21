import os
import json
import numpy as np


# Removes html line breaks from reviews
def remove_breaks(review):
    return review.replace('<br /><br />', '')


# Combines training files into a single file containing all reviews and scores
def create_combined_files(data_type, data_label):
    reviews = []
    scores = []
    for filename in os.listdir(f'data/{data_type}/{data_label}'):
        score = int(filename.split('_')[-1].split('.')[0])
        filepath = f'data/{data_type}/{data_label}/{filename}'
        with open(filepath, 'r', encoding='utf-8') as file:
            review = file.read()
            reviews.append(remove_breaks(review))
            scores.append(score)

    combined_data = {'reviews': reviews, 'scores': scores}

    with open(f'data/combined/{data_type}_{data_label}.json', 'w') as file:
        json.dump(combined_data, file)


# Loads combined files
def load_combined_reviews(data_type, data_label):
    filepath = f'data/combined/{data_type}_{data_label}.json'
    with open(filepath, 'r', encoding='utf-8') as json_file:
        combined_data = json.load(json_file)
    scores = np.array(combined_data['scores'], dtype=np.int8)
    # reduce upper scores by 2 to have continuous range from 1 to 8
    scores[scores > 5] -= 2
    reviews = combined_data['reviews']
    return reviews, scores


def main():
    for data_type in ['train', 'test']:
        for data_label in ['pos', 'neg']:
            create_combined_files(data_type, data_label)


if __name__ == '__main__':
    main()
