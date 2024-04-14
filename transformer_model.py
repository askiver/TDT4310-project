from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
from utils import load_movie_reviews, load_combined_reviews
import numpy as np
from sklearn.metrics import accuracy_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Dataset
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def train_transformer_model():
    # Load the data
    pos_train = load_combined_reviews('train', 'pos')
    neg_train = load_combined_reviews('train', 'neg')
    pos_test = load_combined_reviews('test', 'pos')
    neg_test = load_combined_reviews('test', 'neg')
    train = pos_train + neg_train
    test = pos_test + neg_test
    target_train = [1] * len(pos_train) + [0] * len(neg_train)
    target_test = [1] * len(pos_test) + [0] * len(neg_test)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Tokenize all texts
    train_encodings = tokenizer(train, truncation=True, padding=True, max_length=256)
    test_encodings = tokenizer(test, truncation=True, padding=True, max_length=256)

    # Create datasets
    train_dataset = SentimentDataset(train_encodings, target_train)
    test_dataset = SentimentDataset(test_encodings, target_test)

    # Load pre-trained model with classification head
    model = RobertaForSequenceClassification.from_pretrained('roberta-base').to('cuda' if torch.cuda.is_available() else 'cpu')

    # Training arguments
    training_args = TrainingArguments(
        fp16=True,
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()

    print(eval_results)

    # Save the model
    model.save_pretrained('models/transformer_model')


def main():
    train_transformer_model()


if __name__ == '__main__':
    main()
