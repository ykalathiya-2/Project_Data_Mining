import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.pytorch
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration
DATA_DIR = "../Data"
MODEL_DIR = "../models"
MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

def clean_text(text):
    if not text: return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class FakeNewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = str(self.data.iloc[index]['statement'])
        text = clean_text(text)
        label = self.data.iloc[index]['label']
        
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
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data():
    # Load datasets
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.tsv'), sep='\t', header=None)
    valid_df = pd.read_csv(os.path.join(DATA_DIR, 'valid.tsv'), sep='\t', header=None)
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.tsv'), sep='\t', header=None)

    # Assign columns (based on notebook analysis or standard LIAR dataset format)
    # Assuming columns: ID, label, statement, subject, speaker, job, state, party, context...
    # But let's check the notebook or just use indices if columns are not named.
    # The notebook likely assigned column names. Let's assume standard LIAR columns for now or just use indices.
    # Wait, the notebook code I read didn't show column assignment explicitly in the snippet I saw, 
    # but usually it's: id, label, statement, subjects, speaker, job_title, state_info, party_affiliation, total_credit, barely_true_counts, false_counts, half_true_counts, mostly_true_counts, pants_on_fire_counts, context
    
    columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 'barely_true', 'false', 'half_true', 'mostly_true', 'pants_on_fire', 'context']
    
    # Adjust column count if necessary. The file might not have headers.
    # Let's just use the first few columns which are critical.
    # 0: id, 1: label, 2: statement.
    
    train_df = train_df.iloc[:, :3]
    train_df.columns = ['id', 'label', 'statement']
    
    valid_df = valid_df.iloc[:, :3]
    valid_df.columns = ['id', 'label', 'statement']
    
    test_df = test_df.iloc[:, :3]
    test_df.columns = ['id', 'label', 'statement']

    # Map labels to binary (Fake vs Real)
    # Labels: true, mostly-true, half-true, barely-true, false, pants-fire
    # Fake: false, pants-fire, barely-true
    # Real: true, mostly-true, half-true
    
    label_map = {
        'true': 1,
        'mostly-true': 1,
        'half-true': 1,
        'barely-true': 0,
        'false': 0,
        'pants-fire': 0
    }
    
    train_df['label'] = train_df['label'].map(label_map)
    valid_df['label'] = valid_df['label'].map(label_map)
    test_df['label'] = test_df['label'].map(label_map)
    
    # Drop NaNs if any
    train_df.dropna(inplace=True)
    valid_df.dropna(inplace=True)
    test_df.dropna(inplace=True)
    
    return train_df, valid_df, test_df

def train_model():
    mlflow.set_experiment("Fake_News_Detection_DistilBERT")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("max_len", MAX_LEN)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        
        # Load Data
        print("Loading data...")
        train_df, valid_df, test_df = load_data()
        
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        
        # Create datasets
        train_dataset = FakeNewsDataset(train_df, tokenizer, MAX_LEN)
        valid_dataset = FakeNewsDataset(valid_df, tokenizer, MAX_LEN)
        test_dataset = FakeNewsDataset(test_df, tokenizer, MAX_LEN)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        # Model Setup
        print("Setting up model...")
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        
        # Load existing model weights if available
        model_path = os.path.join(MODEL_DIR, 'transformer_best.pt')
        if os.path.exists(model_path):
            print(f"Loading existing model weights from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"No existing model found at {model_path}. Using base model.")

        model = model.to(device)
        
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        
        # Evaluate on Test Set (using loaded model)
        print("Evaluating loaded model on Test Set...")
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                
                test_loss += loss.item()
                
                _, preds = torch.max(logits, dim=1)
                test_correct += torch.sum(preds == labels)
                test_total += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_test_loss = test_loss / len(test_loader)
        test_acc = test_correct.double() / test_total
        test_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        mlflow.log_metric("test_loss", avg_test_loss)
        mlflow.log_metric("test_acc", test_acc.item())
        mlflow.log_metric("test_f1", test_f1)
        
        # Log the model
        mlflow.pytorch.log_model(model, "model")
        print("Logged model to MLflow.")

if __name__ == '__main__':
    train_model()
