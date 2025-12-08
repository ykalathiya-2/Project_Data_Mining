"""
MLflow Training Pipeline for Fake News Detection
Trains both Transformer (RoBERTa/DistilBERT) and AutoGluon models on LIAR dataset
"""

import os
import re
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm.auto import tqdm
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Paths
    DATA_PATH = '../Data/'
    MODEL_SAVE_PATH = '../models/'
    MLFLOW_TRACKING_URI = 'sqlite:///mlflow.db'
    EXPERIMENT_NAME = 'Fake_News_Detection_Training'
    
    # Transformer settings
    TRANSFORMER_MODEL = 'distilbert-base-uncased'  # or 'roberta-base'
    MAX_LENGTH = 256
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WARMUP_STEPS = 500
    
    # AutoGluon settings
    AG_TIME_LIMIT = 600  # 10 minutes
    AG_PRESET = 'medium_quality'  # 'best_quality', 'high_quality', 'medium_quality'
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Random seed
    SEED = 42

# Set random seeds
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.SEED)

# ============================================================================
# DATA LOADING
# ============================================================================
def load_liar_dataset():
    """Load LIAR dataset from TSV files"""
    print("\n" + "="*70)
    print("📂 LOADING LIAR DATASET")
    print("="*70)
    
    COLUMN_NAMES = [
        'id', 'label', 'statement', 'subject', 'speaker', 'job_title',
        'state', 'party', 'barely_true_count', 'false_count',
        'half_true_count', 'mostly_true_count', 'pants_fire_count', 'context'
    ]
    
    # Load datasets
    train_df = pd.read_csv(f'{Config.DATA_PATH}train.tsv', sep='\t', header=None,
                           names=COLUMN_NAMES, on_bad_lines='skip')
    valid_df = pd.read_csv(f'{Config.DATA_PATH}valid.tsv', sep='\t', header=None,
                           names=COLUMN_NAMES, on_bad_lines='skip')
    test_df = pd.read_csv(f'{Config.DATA_PATH}test.tsv', sep='\t', header=None,
                          names=COLUMN_NAMES, on_bad_lines='skip')
    
    # Binary label mapping
    LABEL_MAP = {
        'true': 1, 'mostly-true': 1, 'half-true': 1,
        'barely-true': 0, 'false': 0, 'pants-fire': 0
    }
    
    for df in [train_df, valid_df, test_df]:
        df['binary_label'] = df['label'].map(LABEL_MAP)
    
    # Drop unmapped labels
    train_df = train_df.dropna(subset=['binary_label'])
    valid_df = valid_df.dropna(subset=['binary_label'])
    test_df = test_df.dropna(subset=['binary_label'])
    
    train_df['binary_label'] = train_df['binary_label'].astype(int)
    valid_df['binary_label'] = valid_df['binary_label'].astype(int)
    test_df['binary_label'] = test_df['binary_label'].astype(int)
    
    print(f"✅ Data loaded successfully!")
    print(f"   Training:   {len(train_df):,} samples")
    print(f"   Validation: {len(valid_df):,} samples")
    print(f"   Test:       {len(test_df):,} samples")
    print(f"   Total:      {len(train_df) + len(valid_df) + len(test_df):,} samples")
    
    return train_df, valid_df, test_df

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================
def clean_text(text):
    """Clean and preprocess text"""
    if not text or pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_data(train_df, valid_df, test_df):
    """Apply text cleaning to all datasets"""
    print("\n" + "="*70)
    print("🧹 PREPROCESSING TEXT DATA")
    print("="*70)
    
    for df in [train_df, valid_df, test_df]:
        df['statement_clean'] = df['statement'].apply(clean_text)
        df['speaker'] = df['speaker'].fillna('unknown')
        df['party'] = df['party'].fillna('none')
        df['state'] = df['state'].fillna('unknown')
    
    print("✅ Text preprocessing completed!")
    return train_df, valid_df, test_df

# ============================================================================
# PYTORCH DATASET CLASS
# ============================================================================
class FakeNewsDataset(Dataset):
    """PyTorch Dataset for Fake News Detection"""
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ============================================================================
# TRANSFORMER TRAINING FUNCTIONS
# ============================================================================
def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy

def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    predictions = []
    true_labels = []
    probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            preds = torch.argmax(outputs.logits, dim=1)
            probs = torch.softmax(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='binary')
    recall = recall_score(true_labels, predictions, average='binary')
    f1 = f1_score(true_labels, predictions, average='binary')
    
    # Calculate ROC AUC
    probabilities = np.array(probabilities)
    roc_auc = roc_auc_score(true_labels, probabilities[:, 1])
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

def train_transformer_model(train_df, valid_df, test_df):
    """Train transformer model with MLflow tracking"""
    print("\n" + "="*70)
    print("🤖 TRAINING TRANSFORMER MODEL")
    print("="*70)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"Transformer_{Config.TRANSFORMER_MODEL}"):
        # Log parameters
        mlflow.log_param("model_type", "transformer")
        mlflow.log_param("model_name", Config.TRANSFORMER_MODEL)
        mlflow.log_param("max_length", Config.MAX_LENGTH)
        mlflow.log_param("batch_size", Config.BATCH_SIZE)
        mlflow.log_param("learning_rate", Config.LEARNING_RATE)
        mlflow.log_param("num_epochs", Config.NUM_EPOCHS)
        mlflow.log_param("train_samples", len(train_df))
        mlflow.log_param("valid_samples", len(valid_df))
        mlflow.log_param("test_samples", len(test_df))
        
        # Initialize tokenizer and model
        print(f"\n📦 Loading {Config.TRANSFORMER_MODEL}...")
        tokenizer = AutoTokenizer.from_pretrained(Config.TRANSFORMER_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            Config.TRANSFORMER_MODEL,
            num_labels=2
        ).to(Config.DEVICE)
        
        # Create datasets
        train_dataset = FakeNewsDataset(
            train_df['statement_clean'].values,
            train_df['binary_label'].values,
            tokenizer,
            Config.MAX_LENGTH
        )
        valid_dataset = FakeNewsDataset(
            valid_df['statement_clean'].values,
            valid_df['binary_label'].values,
            tokenizer,
            Config.MAX_LENGTH
        )
        test_dataset = FakeNewsDataset(
            test_df['statement_clean'].values,
            test_df['binary_label'].values,
            tokenizer,
            Config.MAX_LENGTH
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=Config.BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
        total_steps = len(train_loader) * Config.NUM_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=Config.WARMUP_STEPS,
            num_training_steps=total_steps
        )
        
        # Training loop
        best_valid_acc = 0
        for epoch in range(Config.NUM_EPOCHS):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
            print(f"{'='*70}")
            
            # Train
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, Config.DEVICE)
            print(f"\n📊 Training   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            
            # Validate
            valid_metrics = evaluate(model, valid_loader, Config.DEVICE)
            print(f"📊 Validation - Accuracy: {valid_metrics['accuracy']:.4f}, "
                  f"F1: {valid_metrics['f1']:.4f}, AUC: {valid_metrics['roc_auc']:.4f}")
            
            # Log metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("valid_accuracy", valid_metrics['accuracy'], step=epoch)
            mlflow.log_metric("valid_f1", valid_metrics['f1'], step=epoch)
            mlflow.log_metric("valid_roc_auc", valid_metrics['roc_auc'], step=epoch)
            
            # Save best model
            if valid_metrics['accuracy'] > best_valid_acc:
                best_valid_acc = valid_metrics['accuracy']
                model_path = os.path.join(Config.MODEL_SAVE_PATH, 'transformer_best.pt')
                os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
                torch.save(model.state_dict(), model_path)
                print(f"✅ Best model saved! (Accuracy: {best_valid_acc:.4f})")
        
        # Final test evaluation
        print(f"\n{'='*70}")
        print("🧪 FINAL TEST EVALUATION")
        print(f"{'='*70}")
        test_metrics = evaluate(model, test_loader, Config.DEVICE)
        
        print(f"\n📊 Test Results:")
        print(f"   Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"   Precision: {test_metrics['precision']:.4f}")
        print(f"   Recall:    {test_metrics['recall']:.4f}")
        print(f"   F1 Score:  {test_metrics['f1']:.4f}")
        print(f"   ROC AUC:   {test_metrics['roc_auc']:.4f}")
        
        # Log final test metrics
        mlflow.log_metric("test_accuracy", test_metrics['accuracy'])
        mlflow.log_metric("test_precision", test_metrics['precision'])
        mlflow.log_metric("test_recall", test_metrics['recall'])
        mlflow.log_metric("test_f1", test_metrics['f1'])
        mlflow.log_metric("test_roc_auc", test_metrics['roc_auc'])
        
        # Log model
        mlflow.pytorch.log_model(model, "model")
        
        print("\n✅ Transformer training completed!")
        return test_metrics

# ============================================================================
# AUTOGLUON TRAINING
# ============================================================================
def train_autogluon_model(train_df, valid_df, test_df):
    """Train AutoGluon model with MLflow tracking"""
    print("\n" + "="*70)
    print("🤖 TRAINING AUTOGLUON MODEL")
    print("="*70)
    
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        print("⚠️ AutoGluon not installed. Skipping AutoGluon training.")
        print("   Install with: pip install autogluon")
        return None
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"AutoGluon_{Config.AG_PRESET}"):
        # Log parameters
        mlflow.log_param("model_type", "autogluon")
        mlflow.log_param("preset", Config.AG_PRESET)
        mlflow.log_param("time_limit", Config.AG_TIME_LIMIT)
        mlflow.log_param("train_samples", len(train_df))
        mlflow.log_param("valid_samples", len(valid_df))
        mlflow.log_param("test_samples", len(test_df))
        
        # Prepare data for AutoGluon
        train_data = train_df[['statement_clean', 'speaker', 'party', 'state',
                                'barely_true_count', 'false_count', 'half_true_count',
                                'mostly_true_count', 'pants_fire_count', 'binary_label']].copy()
        
        test_data = test_df[['statement_clean', 'speaker', 'party', 'state',
                              'barely_true_count', 'false_count', 'half_true_count',
                              'mostly_true_count', 'pants_fire_count', 'binary_label']].copy()
        
        # Fill NaN values
        for col in ['barely_true_count', 'false_count', 'half_true_count',
                    'mostly_true_count', 'pants_fire_count']:
            train_data[col] = train_data[col].fillna(0).astype(int)
            test_data[col] = test_data[col].fillna(0).astype(int)
        
        # Train AutoGluon
        print(f"\n📦 Training AutoGluon with preset: {Config.AG_PRESET}")
        print(f"   Time limit: {Config.AG_TIME_LIMIT} seconds")
        
        save_path = os.path.join(Config.MODEL_SAVE_PATH, 'autogluon_model')
        os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
        
        predictor = TabularPredictor(
            label='binary_label',
            path=save_path,
            eval_metric='accuracy',
            problem_type='binary'
        ).fit(
            train_data=train_data,
            time_limit=Config.AG_TIME_LIMIT,
            presets=Config.AG_PRESET,
            verbosity=2
        )
        
        # Evaluate on test set
        print(f"\n{'='*70}")
        print("🧪 EVALUATING ON TEST SET")
        print(f"{'='*70}")
        
        y_pred = predictor.predict(test_data)
        y_pred_proba = predictor.predict_proba(test_data)
        y_true = test_data['binary_label']
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        
        # ROC AUC
        if hasattr(y_pred_proba, 'values'):
            proba_positive = y_pred_proba[1].values if 1 in y_pred_proba.columns else y_pred_proba.iloc[:, 1].values
        else:
            proba_positive = y_pred_proba[:, 1]
        roc_auc = roc_auc_score(y_true, proba_positive)
        
        test_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
        
        print(f"\n📊 Test Results:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1 Score:  {f1:.4f}")
        print(f"   ROC AUC:   {roc_auc:.4f}")
        
        # Log metrics
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_roc_auc", roc_auc)
        
        # Get leaderboard
        leaderboard = predictor.leaderboard(test_data, silent=True)
        print("\n📋 Model Leaderboard (Top 5):")
        print(leaderboard.head())
        
        # Log leaderboard
        leaderboard_path = 'autogluon_leaderboard.csv'
        leaderboard.to_csv(leaderboard_path, index=False)
        mlflow.log_artifact(leaderboard_path)
        os.remove(leaderboard_path)
        
        # Log model path (AutoGluon models are too large to log directly)
        mlflow.log_param("model_save_path", save_path)
        
        print("\n✅ AutoGluon training completed!")
        return test_metrics
    
# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("🚀 FAKE NEWS DETECTION - MLFLOW TRAINING PIPELINE")
    print("="*70)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  Device: {Config.DEVICE}")
    print(f"🔢 Random Seed: {Config.SEED}")
    
    # Setup MLflow
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.EXPERIMENT_NAME)
    
    print(f"\n📊 MLflow Tracking URI: {Config.MLFLOW_TRACKING_URI}")
    print(f"🧪 Experiment: {Config.EXPERIMENT_NAME}")
    
    # Load data
    train_df, valid_df, test_df = load_liar_dataset()
    
    # Preprocess data
    train_df, valid_df, test_df = preprocess_data(train_df, valid_df, test_df)
    
    # Train Transformer Model
    transformer_metrics = train_transformer_model(train_df, valid_df, test_df)
    
    # Train AutoGluon Model
    autogluon_metrics = train_autogluon_model(train_df, valid_df, test_df)
    
    # Summary
    print("\n" + "="*70)
    print("📊 TRAINING SUMMARY")
    print("="*70)
    
    print("\n🤖 Transformer Model:")
    if transformer_metrics:
        for metric, value in transformer_metrics.items():
            print(f"   {metric.capitalize()}: {value:.4f}")
    
    print("\n🤖 AutoGluon Model:")
    if autogluon_metrics:
        for metric, value in autogluon_metrics.items():
            print(f"   {metric.capitalize()}: {value:.4f}")
    else:
        print("   ⚠️ AutoGluon training was skipped")
    
    print("\n" + "="*70)
    print(f"✅ Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print(f"\n💡 To view results, run: mlflow ui --backend-store-uri {Config.MLFLOW_TRACKING_URI}")
    print(f"   Then open: http://localhost:5000")

if __name__ == "__main__":
    main()
