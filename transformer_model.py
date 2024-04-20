from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
from utils import load_movie_reviews, load_combined_reviews
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, classification_report, \
    precision_recall_fscore_support


def compute_metrics_binary(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, _, _ = precision_recall_fscore_support(labels, predictions, average=None, labels=[0, 1])

    # Assuming binary classification, index 0 is the negative class, and index 1 is the positive class
    result = {
        "accuracy": accuracy,
        "positive_precision": precision[1],
        "positive_recall": recall[1],
        "negative_precision": precision[0],
        "negative_recall": recall[0]
    }
    return result


def compute_metrics_regression(eval_pred):
    logits, labels = eval_pred
    predictions = np.round(logits)
    # set min value to 1 and max value to 8
    predictions[predictions < 1] = 1
    predictions[predictions > 8] = 8
    result = {"accuracy": accuracy_score(labels, predictions),
              "mean_squared_error": mean_squared_error(labels, predictions),
              "mean_absolute_error": mean_absolute_error(labels, predictions)}
    return result


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


def train_transformer_model(binary_classification=True):
    # Load the data
    pos_train, pos_train_scores = load_combined_reviews('train', 'pos')
    neg_train, neg_train_scores = load_combined_reviews('train', 'neg')
    pos_test, pos_test_scores = load_combined_reviews('test', 'pos')
    neg_test, neg_test_scores = load_combined_reviews('test', 'neg')
    train = pos_train + neg_train
    test = pos_test + neg_test
    if binary_classification:
        target_train = [1] * len(pos_train) + [0] * len(neg_train)
        target_test = [1] * len(pos_test) + [0] * len(neg_test)
    else:
        target_train = np.concatenate((np.array(pos_train_scores), np.array(neg_train_scores)), dtype=np.float32)
        target_test = np.concatenate((np.array(pos_test_scores), np.array(neg_test_scores)), dtype=np.float32)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Tokenize all texts
    train_encodings = tokenizer(train, truncation=True, padding=True, max_length=256)
    test_encodings = tokenizer(test, truncation=True, padding=True, max_length=256)

    # Create datasets
    train_dataset = SentimentDataset(train_encodings, target_train)
    test_dataset = SentimentDataset(test_encodings, target_test)

    if binary_classification:
        num_labels = 2
    else:
        num_labels = 1

    # Load pre-trained model with classification head
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels).to(
        'cuda' if torch.cuda.is_available() else 'cpu')

    compute_metrics = compute_metrics_binary if binary_classification else compute_metrics_regression

    # Training arguments
    training_args = TrainingArguments(
        fp16=True,
        output_dir='./results',
        num_train_epochs=10,
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


def test_model(binary_classification=True):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_labels = 2 if binary_classification else 1
    model  = RobertaForSequenceClassification.from_pretrained('models/transformer_model', num_labels=num_labels).to(
        'cuda' if torch.cuda.is_available() else 'cpu')

    #model.load_state_dict(torch.load(f'models/transformer_model/config.json'))

    # Load test set
    pos_test, pos_test_scores = load_combined_reviews('test', 'pos')
    neg_test, neg_test_scores = load_combined_reviews('test', 'neg')
    test = pos_test + neg_test
    if binary_classification:
        target_test = [1] * len(pos_test) + [0] * len(neg_test)
    else:
        target_test = np.concatenate((np.array(pos_test_scores), np.array(neg_test_scores)), dtype=np.float32)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Tokenize all texts
    test_encodings = tokenizer(test, truncation=True, padding=True, max_length=256)

    # Create datasets
    test_dataset = SentimentDataset(test_encodings, target_test)

    MAE = 0
    test_accuracy = 0
    all_outputs = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for data in test_dataset:
            features, target = data['input_ids'].to(DEVICE), data['labels'].to(
                DEVICE)  # Assuming input_ids and labels are keys
            outputs = model(features)
            if binary_classification:
                predictions = torch.argmax(outputs, dim=1)
                test_accuracy += (predictions == target).sum().item()
            else:
                outputs = model.forward_score(features)
                test_accuracy += outputs.eq(target).sum().item()
                MAE += mean_absolute_error(target.cpu().numpy(), outputs.cpu().numpy())

    print(f"Test accuracy: {test_accuracy / len(test_dataset.dataset):.4f}")

    if binary_classification:
        print(f"classification report: {classification_report(all_targets, all_outputs, digits=4)}")


def main():
    train_transformer_model(binary_classification=False)
    #test_model(binary_classification=True)


if __name__ == '__main__':
    main()
