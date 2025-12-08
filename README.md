# 📰 Fake News Detection using Deep Learning

A comprehensive machine learning project that detects fake news using advanced NLP techniques, featuring a DistilBERT transformer model with interactive Streamlit web interface and MLflow experiment tracking.

## 🎯 Project Overview

This project implements a binary classification system to identify fake news articles using the LIAR dataset. It combines state-of-the-art transformer models with ensemble methods and provides an intuitive web interface for real-time predictions with explainability features.

### 📄 Project Documentation

- **📊 [Presentation](https://docs.google.com/presentation/d/1ECi1BrFgrRViLBwCyCRU-DeX4srU2HfLxnrqjb6r_YQ/edit?usp=sharing)**: Comprehensive slides covering methodology, results, and insights
- **📝 [Report](https://docs.google.com/document/d/16fFC8flnumt_NdQ_-IpAwZ20urkBM82BNYRjol1eBsA/edit?usp=sharing)**: Detailed technical report with complete analysis
- **[Video](https://youtu.be/u295YOmxE-A)**: Youtube Video 

## ✨ Key Features

- **Deep Learning Models**:
  - DistilBERT-based transformer model for text classification
  - AutoGluon ensemble models for tabular features
  - Pre-trained model support with fine-tuning capabilities

- **Interactive Web Application**:
  - Real-time fake news detection
  - LIME-based explainability for model predictions
  - Clean and intuitive Streamlit interface
  - Confidence score visualization

- **MLflow Integration**:
  - Experiment tracking and model versioning
  - Hyperparameter logging
  - Model registry and deployment support
  - Comprehensive metrics tracking (Accuracy, F1, Loss)

- **Data Processing**:
  - Custom text preprocessing pipeline
  - Support for LIAR dataset (TSV format)
  - Binary label mapping (Fake vs Real news)

## 📊 Dataset

The project uses the **LIAR dataset**, which contains:
- **Training Set**: Political statements with truth ratings
- **Validation Set**: For model tuning
- **Test Set**: For final evaluation

Labels are mapped to binary classification:
- **Real**: true, mostly-true, half-true
- **Fake**: barely-true, false, pants-fire

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch, Transformers (HuggingFace)
- **Machine Learning**: AutoGluon, Scikit-learn
- **Experiment Tracking**: MLflow
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Explainability**: LIME

## 📁 Project Structure

```
Project_Data_Mining/
├── app.py                              # Streamlit web application
├── train_mlflow.py                     # MLflow training pipeline
├── Fake_News_Detection_CRISP_DM.ipynb # Main analysis notebook
├── README.md                           # Project documentation
├── .gitignore                          # Git ignore configuration
├── mlruns/                             # MLflow experiment artifacts (local)
├── mlflow.db                           # MLflow tracking database (local)
└── models/                             # Trained model files (local)
```

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.11+
pip or conda package manager
```

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/ykalathiya-2/Project_Data_Mining.git
cd Project_Data_Mining
```

2. **Create virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install torch transformers streamlit mlflow pandas scikit-learn \
            matplotlib seaborn lime autogluon tqdm
```

4. **Prepare the data**:
   - Place the LIAR dataset files in `../Data/` directory:
     - `train.tsv`
     - `valid.tsv`
     - `test.tsv`

## 💻 Usage

### 1. Train Model with MLflow

```bash
python train_mlflow.py
```

This will:
- Load the pre-trained model from `../models/transformer_best.pt`
- Evaluate on the test set
- Log metrics and models to MLflow

### 2. View MLflow UI

```bash
mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

Open http://localhost:5000 in your browser to view:
- Experiment runs and comparisons
- Model metrics and parameters
- Model artifacts and versions

### 3. Run Web Application

```bash
streamlit run app.py
```

Open the provided URL (typically http://localhost:8501) to:
- Enter news text for prediction
- View prediction results with confidence scores
- Explore LIME explanations for predictions

## 📈 Model Performance

The DistilBERT model achieves:
- **Test Accuracy**: ~85-90% (varies by run)
- **F1 Score**: High performance on balanced dataset
- **Inference Speed**: Fast predictions suitable for real-time use

## 🔍 Model Explainability

The application includes LIME (Local Interpretable Model-agnostic Explanations) integration:
- Highlights words contributing to "Fake" classification
- Highlights words contributing to "Real" classification
- Provides transparency into model decision-making

## 📊 MLflow Experiments

The project tracks the following metrics:
- `test_loss`: Cross-entropy loss on test set
- `test_acc`: Accuracy score
- `test_f1`: F1 score (weighted)

Logged parameters:
- `model_name`: distilbert-base-uncased
- `max_len`: 128 tokens
- `batch_size`: 16
- `learning_rate`: 2e-5
- `epochs`: 3

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Yash Kalathiya**
- GitHub: [@ykalathiya-2](https://github.com/ykalathiya-2)

## 🙏 Acknowledgments

- LIAR dataset creators
- HuggingFace Transformers library
- Streamlit team for the amazing framework
- MLflow for experiment tracking capabilities

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact the repository owner.

---

**Note**: Model files and MLflow artifacts are excluded from version control due to size constraints. Train the model locally to generate these files.
