import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import metrics
from utils import remove_breaks, load_combined_reviews
from sklearn.utils import shuffle


def train_bayes_model(binary_classification=True):
    pos_train, scores_train_pos = load_combined_reviews('train', 'pos')
    neg_train, scores_train_neg = load_combined_reviews('train', 'neg')
    pos_test, scores_test_pos = load_combined_reviews('test', 'pos')
    neg_test, scores_test_neg = load_combined_reviews('test', 'neg')

    train = np.concatenate((pos_train, neg_train))
    test = np.concatenate((pos_test, neg_test))

    if binary_classification:
        target_train = [1] * len(pos_train) + [0] * len(neg_train)
        target_test = [1] * len(pos_test) + [0] * len(neg_test)
    else:
        target_train = np.concatenate((scores_train_pos, scores_train_neg))
        target_test = np.concatenate((scores_test_pos, scores_test_neg))

    train, target_train = shuffle(train, target_train, random_state=0)
    test, target_test = shuffle(test, target_test, random_state=0)

    vectorizer = TfidfVectorizer(stop_words='english', lowercase=False)
    X_train = vectorizer.fit_transform(train)
    X_test = vectorizer.transform(test)

    model = MultinomialNB()

    model.fit(X_train, target_train)
    pred = model.predict(X_test)

    if binary_classification:
        print(metrics.classification_report(target_test, pred, digits=4))
    else:
        # set min value to 1 and max value to 8
        pred[pred < 1] = 1
        pred[pred > 8] = 8
        # Round predictions to nearest integer
        pred = np.round(pred)

        print(metrics.mean_squared_error(target_test, pred))
        print(metrics.mean_absolute_error(target_test, pred))
        # Find number of correct guesses
        print(metrics.accuracy_score(target_test, pred))


def main():
    train_bayes_model(binary_classification=False)


if __name__ == '__main__':
    main()
