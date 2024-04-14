import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from utils import remove_breaks
from sklearn.utils import shuffle


def load_movie_reviews(folder):
    reviews = []
    for filename in os.listdir(folder):
        filepath = f'{folder}/{filename}'
        with open(filepath, 'r', encoding='utf-8') as file:
            review = file.read()
            reviews.append(remove_breaks(review))
    return reviews


def train_bayes_model():
    pos_train = load_movie_reviews('data/train/pos')
    neg_train = load_movie_reviews('data/train/neg')
    pos_test = load_movie_reviews('data/test/pos')
    neg_test = load_movie_reviews('data/test/neg')

    train = pos_train + neg_train
    test = pos_test + neg_test
    target_train = [1] * len(pos_train) + [0] * len(neg_train)
    target_test = [1] * len(pos_test) + [0] * len(neg_test)

    train, target_train = shuffle(train, target_train, random_state=0)
    test, target_test = shuffle(test, target_test, random_state=0)

    #vectorizer = CountVectorizer()
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=False)
    X_train = vectorizer.fit_transform(train)
    X_test = vectorizer.transform(test)

    clf = MultinomialNB()
    clf.fit(X_train, target_train)
    pred = clf.predict(X_test)

    print(metrics.classification_report(target_test, pred))







def main():
    train_bayes_model()


if __name__ == '__main__':
    main()
