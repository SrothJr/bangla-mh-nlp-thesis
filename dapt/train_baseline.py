import os
import pandas as pd
import numpy as np
import torch
import warnings
import json
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from utils import compute_metrics, plot_confusion_matrix, plot_performance

# Ignore specific huggingface warnings for cleaner output
warnings.filterwarnings("ignore")

# Directories
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw/dataset.xlsx"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/baselines"))
FIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/figures"))

MODELS = {
    "BanglaBERT": "csebuetnlp/banglabert",
    "sahajBERT": "neuropark/sahajBERT",
    "mBERT": "bert-base-multilingual-cased"
}

CLASSES = ["Minimum", "Mild", "Moderate", "Severe"]
ID2LABEL = {i: c for i, c in enumerate(CLASSES)}
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}

class BanglaDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH)
    
    print("Using 'posts' for text and 'labels' for labels.")
    
    texts = df['posts'].astype(str).tolist()
    labels_raw = df['labels'].tolist()

    # Convert string labels to numerical IDs, and 1-indexed to 0-indexed if needed
    labels = []
    for l in labels_raw:
        if isinstance(l, str):
            labels.append(LABEL2ID.get(l.capitalize(), 0))
        else:
            # Convert 1-indexed labels (1, 2, 3, 4) to 0-indexed (0, 1, 2, 3)
            labels.append(int(l) - 1)

    # Stratified 5-Fold Cross Validation Setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    final_results = {}

    for model_name, model_id in MODELS.items():
        print(f"\n{'='*60}\nTraining Base Model: {model_name} ({model_id})\n{'='*60}")
        
        metrics_file = os.path.join(OUTPUT_DIR, f"{model_name}_metrics.json")
        if os.path.exists(metrics_file):
            print(f"Skipping {model_name} because it is already trained. Loading saved metrics.")
            with open(metrics_file, 'r') as f:
                final_results[model_name] = json.load(f)
            continue
            
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        all_y_true = []
        all_y_pred = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
            print(f"\n--- Fold {fold + 1}/5 ---")
            
            train_texts = [texts[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_texts = [texts[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]

            # Thesis uses 256 for fine-tuning
            train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=256)
            val_encodings = tokenizer(val_texts, truncation=True, padding='max_length', max_length=256)

            train_dataset = BanglaDataset(train_encodings, train_labels)
            val_dataset = BanglaDataset(val_encodings, val_labels)

            model = AutoModelForSequenceClassification.from_pretrained(
                model_id, 
                num_labels=4, 
                id2label=ID2LABEL, 
                label2id=LABEL2ID
            )

            # Training arguments with Early Stopping
            training_args = TrainingArguments(
                output_dir=os.path.join(OUTPUT_DIR, f"{model_name}_fold{fold+1}"),
                num_train_epochs=15,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=32,
                learning_rate=2e-5,
                weight_decay=0.01,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                save_total_limit=1,
                logging_steps=50,
                seed=42,
                fp16=torch.cuda.is_available(),
                report_to="none" # Disable wandb etc.
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )

            trainer.train()

            # Predict and accumulate for overall metrics and confusion matrix
            preds = trainer.predict(val_dataset)
            predictions = preds.predictions[0] if isinstance(preds.predictions, tuple) else preds.predictions
            y_pred = predictions.argmax(-1)
            
            all_y_true.extend(val_labels)
            all_y_pred.extend(y_pred)

        # Calculate exact overall metrics
        overall_metrics = compute_metrics(type('obj', (object,), {'label_ids': np.array(all_y_true), 'predictions': np.eye(4)[all_y_pred]})())
        final_results[model_name] = overall_metrics
        
        with open(metrics_file, 'w') as f:
            json.dump(overall_metrics, f)
        
        print(f"\n[DONE] {model_name} Overall 5-Fold Results:")
        print(f"Accuracy: {overall_metrics['accuracy']:.4f}, F1: {overall_metrics['f1']:.4f}")
        
        # Output High-Res Confusion Matrix
        plot_confusion_matrix(
            all_y_true, 
            all_y_pred, 
            CLASSES, 
            f'Base {model_name} Confusion Matrix', 
            os.path.join(FIG_DIR, f'cm_base_{model_name.lower()}.png')
        )

    # Plot final aggregated performance bar chart
    plot_performance(final_results, os.path.join(FIG_DIR, 'baseline_performance_comparison.png'))
    print(f"\nAll models finished! Metrics and graphs are saved in {FIG_DIR}.")

if __name__ == "__main__":
    main()
