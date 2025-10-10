import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('track-a.csv')

# Data Exploration
print("Initial data shape:", df.shape)
print("\nEmotion distribution:")
print(df[['anger', 'fear', 'joy', 'sadness', 'surprise']].sum())


# Combine emotion columns into a single label
emotion_cols = ['anger', 'fear', 'joy', 'sadness', 'surprise']
df['labels'] = df[emotion_cols].values.tolist()

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

print(f"\nFinal dataset sizes:")
print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(val_df)}")
print(f"Test size: {len(test_df)}")

# DistilBERT Tokenizer and Model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(emotion_cols),
    problem_type="multi_label_classification"
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Dataset Class
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }

# Parameters
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4

# Data loaders
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = EmotionDataset(
        texts=df.text.values,
        labels=df.labels.values,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=4)

train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(val_df, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)

# Training Setup
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss_fn = torch.nn.BCEWithLogitsLoss().to(device)

# Training Function
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model.train()
    losses = []
    correct_predictions = 0
    for d in tqdm(data_loader, desc="Training"):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        preds = torch.sigmoid(logits) > 0.5
        correct_predictions += torch.sum(preds == labels.byte()).item() / len(emotion_cols)
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions / n_examples, np.mean(losses)

# Evaluation Function
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            preds = torch.sigmoid(logits) > 0.5
            correct_predictions += torch.sum(preds == labels.byte()).item() / len(emotion_cols)
            losses.append(loss.item())
    return correct_predictions / n_examples, np.mean(losses)

# Training Loop
history = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}
best_accuracy = 0

for epoch in range(EPOCHS):
    print(f'\nEpoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(train_df))
    print(f'Train loss {train_loss:.4f} accuracy {train_acc:.4f}')
    val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device, len(val_df))
    print(f'Validation loss {val_loss:.4f} accuracy {val_acc:.4f}')
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model_state_distilbert.bin')
        best_accuracy = val_acc

# Test Evaluation
model.load_state_dict(torch.load('best_model_state_distilbert.bin'))
test_acc, test_loss = eval_model(model, test_data_loader, loss_fn, device, len(test_df))
print(f'\nTest Accuracy: {test_acc:.4f}')
print(f'Test Loss: {test_loss:.4f}')

# Get predictions for classification report
def get_predictions(model, data_loader):
    model.eval()
    texts = []
    predictions = []
    prediction_probs = []
    real_values = []
    with torch.no_grad():
        for d in tqdm(data_loader, desc="Getting predictions"):
            texts.extend(d["text"])
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.sigmoid(logits) > 0.5
            predictions.extend(preds)
            prediction_probs.extend(torch.sigmoid(logits))
            real_values.extend(labels)
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return texts, predictions, prediction_probs, real_values

y_texts, y_pred, y_pred_probs, y_test = get_predictions(model, test_data_loader)

# classification report
from sklearn.metrics import classification_report

print("\nClassification Report:")
report = classification_report(y_test.numpy(), y_pred.numpy(), target_names=emotion_cols, digits=3, output_dict=True)

# Print per-class metrics
print(f"{'':<10}| {'precision':<10} | {'recall':<10} | {'f1-score':<10} | {'support':<10}")
print("-" * 55)
for emotion in emotion_cols:
    print(f"{emotion:<10}| {report[emotion]['precision']:<10.3f} | {report[emotion]['recall']:<10.3f} | "
          f"{report[emotion]['f1-score']:<10.3f} | {int(report[emotion]['support']):<10}")

# Print averages
print("\nmicro avg")
print(f"{report['micro avg']['precision']:.3f}")
print(f"{report['micro avg']['recall']:.3f}")
print(f"{report['micro avg']['f1-score']:.3f}")
print(f"{int(report['micro avg']['support'])}")

print("\nmacro avg")
print(f"{report['macro avg']['precision']:.3f}")
print(f"{report['macro avg']['recall']:.3f}")
print(f"{report['macro avg']['f1-score']:.3f}")
print(f"{int(report['macro avg']['support'])}")

print("\nweighted avg")
print(f"{report['weighted avg']['precision']:.3f}")
print(f"{report['weighted avg']['recall']:.3f}")
print(f"{report['weighted avg']['f1-score']:.3f}")
print(f"{int(report['weighted avg']['support'])}")

print("\nsamples avg")
print(f"{report['samples avg']['precision']:.3f}")
print(f"{report['samples avg']['recall']:.3f}")
print(f"{report['samples avg']['f1-score']:.3f}")
print(f"{int(report['samples avg']['support'])}")

print(f"\nMacro-F1: {report['macro avg']['f1-score']}")

