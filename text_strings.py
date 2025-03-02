import torch
from datasets import Dataset
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Example tokenized data
sentences = ["A class-action lawsuit has been filed against POSCO."]
labels = [["O", "O", "O", "O", "O", "O", "O", "O", "O", "B-ORG", "I-ORG", "O"]]  # Word-aligned BIO labels

# Label mapping
label_map = {"O": 0, "B-ORG": 1, "I-ORG": 2}

# Tokenize and align labels
def tokenize_and_align_labels(sentence, label_list):
    tokenized = tokenizer(sentence, truncation=True, padding="max_length", return_tensors="pt", is_split_into_words=False)
    word_ids = tokenized.word_ids(batch_index=0)  # Align tokens with original words
    label_ids = []

    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:  # CLS/SEP tokens
            label_ids.append(-100)  # Ignore in loss function
        elif word_idx != previous_word_idx:  # First subword of a word
            label_ids.append(label_map[label_list[word_idx]])
        else:  # Subsequent subwords
            label_ids.append(label_map[label_list[word_idx]] if label_list[word_idx] != "O" else 0)  # O-labels stay O
        
        previous_word_idx = word_idx

    tokenized["labels"] = torch.tensor(label_ids)
    return tokenized

# Convert dataset
tokenized_datasets = [tokenize_and_align_labels(sent, lbl) for sent, lbl in zip(sentences, labels)]

# Convert to Hugging Face dataset format
dataset = Dataset.from_list(tokenized_datasets)

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

# Load DistilBERT for token classification (NER)
model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_map))

# Define training arguments
training_args = TrainingArguments(
    output_dir="./ner_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,  # Use a real validation split in practice
    tokenizer=tokenizer
)

trainer.train()
