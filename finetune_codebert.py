import torch
import pandas as pd
from datasets import Dataset
import evaluate
from transformers.models.auto import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from typing import Tuple
import os
import shutil
import numpy as np

MODEL_CHECKPOINT = "microsoft/codebert-base"
MODEL_OUTPUT_DIR = "./codebert_dce_model"
DATASET_PATH = "dce_synthetic_dataset.json"
MAX_LENGTH = 128
TRAINING_EPOCHS = 5

def load_dataset() -> Tuple[Dataset, Dataset]:
    print(f"Loading dataset from existing file: '{DATASET_PATH}'...")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found at '{DATASET_PATH}'.")
    df = None
    try:
        df = pd.read_json(DATASET_PATH, orient='records', lines=True)
        print("Successfully loaded dataset in JSON Lines format.")
    except ValueError:
        try:
            df = pd.read_json(DATASET_PATH, orient='records', lines=False)
            print("Successfully loaded dataset in standard JSON array format.")
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            raise FileNotFoundError("Could not load dataset.")
    if 'code_snippet' in df.columns:
        df = df.rename(columns={'code_snippet': 'text'})
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")
    df['label'] = df['label'].astype(int)
    eval_size = max(5, len(df) // 20)
    split_index = len(df) - eval_size
    train_df = df.iloc[:split_index].reset_index(drop=True)
    eval_df = df.iloc[split_index:].reset_index(drop=True)
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    return train_dataset, eval_dataset

def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def fine_tune_model():
    print("--- 1. Loading and Preparing External Dataset ---")
    train_dataset, eval_dataset = load_dataset()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    print(f"Train dataset size: {len(tokenized_train_dataset)}")
    print(f"Evaluation dataset size: {len(tokenized_eval_dataset)}")
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(["text"])
    tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(["text"])
    print("\n--- 2. Loading Model and Defining Trainer ---")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=2, use_safetensors=True)
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        num_train_epochs=TRAINING_EPOCHS,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    print("\n--- 3. Starting Fine-Tuning (5 Epochs) ---")
    trainer.train()
    print("\n--- 4. Saving Model ---")
    if os.path.exists(MODEL_OUTPUT_DIR):
        print(f"Cleaning existing output directory: {MODEL_OUTPUT_DIR}")
        try:
            shutil.rmtree(MODEL_OUTPUT_DIR)
        except Exception as e:
            print(f"Warning: Could not fully delete directory. Error: {e}")
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"Model and Tokenizer saved to: {MODEL_OUTPUT_DIR}")

if __name__ == "__main__":
    fine_tune_model()
