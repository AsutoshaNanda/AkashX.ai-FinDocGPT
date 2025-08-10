# FinDocGPT - AI-Powered Financial Analysis Platform

**Advanced AI-Powered Financial Document Analysis System**

FinDocGPT transforms how financial professionals interact with dense financial documents like 10-K reports, quarterly filings, and earnings statements. Instead of spending hours manually analyzing hundreds of pages, users can ask natural language questions and get precise, confidence-scored answers within seconds.

## 🚀 Project Overview

FinDocGPT is a comprehensive AI platform that integrates multiple cutting-edge models to provide:

- **Advanced Question-Answering System** using RoBERTa models fine-tuned for financial documents
- **Financial Sentiment Analysis** with specialized FinBERT models
- **Intelligent Document Retrieval** using FAISS indexing and semantic search
- **Financial Anomaly Detection** using isolation forests and statistical methods
- **Interactive Web Interface** built with Streamlit for seamless user experience


### Key Features

- Process real financial documents from the FinanceBench dataset
- Natural language querying with confidence scoring
- Multi-model AI analysis combining QA, sentiment, and anomaly detection
- Real-time market data integration
- Production-ready web interface
- Comprehensive financial intelligence system


## 🛠️ Technical Architecture

```
Financial Documents (PDFs) → Document Processor → Intelligent Chunking
                                                          ↓
User Query → Retrieval System (FAISS) → Context Selection → QA System (RoBERTa)
                     ↓                                            ↓
            Sentiment Analysis (FinBERT) ← Combined Results → Answer + Confidence
                     ↓                                            ↓
            Anomaly Detection (Isolation Forest) ← Streamlit Interface → User
```


## 📋 Prerequisites

- Python 3.8 or higher
- 8GB+ RAM recommended for model loading
- Internet connection for downloading pre-trained models


## ⚡ Quick Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd findocgpt
```


### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```


### 3. Install Dependencies

```bash
pip install -r requirements.txt
```


### 4. Download Required NLTK Data

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```


### 5. Run the Application

```bash
streamlit run enhanced_streamlit_app.py
```

The application will be available at `http://localhost:8501`

## 📦 Dependencies

### Core Requirements[^1]

```
streamlit>=1.28.0
plotly>=5.15.0
yfinance>=0.2.18
pandas>=2.0.0
numpy>=1.24.0
transformers>=4.30.0
torch>=2.0.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
scikit-learn>=1.3.0
```


### AI Models Used

| **Component** | **Technology** | **Purpose** |
| :-- | :-- | :-- |
| **Question Answering** | RoBERTa (deepset/roberta-base-squad2) | Financial document queries |
| **Sentiment Analysis** | FinBERT (ProsusAI/finbert) | Financial sentiment scoring |
| **Document Retrieval** | FAISS + Sentence Transformers | Semantic search |
| **Anomaly Detection** | Isolation Forest + Statistical Methods | Financial anomaly identification |
| **Text Processing** | NLTK + Custom Processors | Document chunking |
| **Web Interface** | Streamlit + Plotly | Interactive dashboard |

## 🗂️ Project Structure

```
findocgpt/
├── enhanced_streamlit_app.py      # Main Streamlit application
├── qa_system.ipynb               # Question-answering system[^1]
├── sentiment_analyzer.ipynb      # Financial sentiment analysis[^4]
├── retrieval_system.ipynb        # Document retrieval system[^3]
├── anomaly_detector.ipynb        # Financial anomaly detection[^7]
├── data_processor.ipynb          # Document processing utilities[^8]
├── pdf_processor.ipynb           # PDF text extraction[^11]
├── financebench_app.ipynb        # Flask API backend[^10]
├── test_api.ipynb                # API testing utilities[^6]
├── requirements.txt              # Python dependencies[^2]
├── streamlit_companies_data.json # Company metadata[^5]
└── README.md                     # This file
```


## 🚀 Usage

### Web Interface

1. **Launch the Application**: Run `streamlit run enhanced_streamlit_app.py`
2. **Select Company**: Choose from 40+ companies in the dataset[^2]
3. **Ask Questions**: Use natural language to query financial information
4. **View Results**: Get answers with confidence scores and source attribution
5. **Analyze Sentiment**: View sentiment trends across financial documents
6. **Detect Anomalies**: Identify unusual patterns in financial metrics

### API Usage

The system also provides a Flask API for programmatic access:[^3]

```python
import requests

# Initialize the system
response = requests.post("http://localhost:5000/initialize", json={
    "pdfs_dir": "pdfs",
    "data_dir": "data"
})

# Ask a question
response = requests.post("http://localhost:5000/ask", json={
    "question": "What was Apple's revenue in 2022?",
    "company": "Apple",
    "top_k": 5
})
```


## 📊 Dataset

FinDocGPT uses the **FinanceBench dataset**, which includes:

- **40+ companies** across multiple sectors
- **10-K annual reports** and **10-Q quarterly filings**
- **Real financial documents** from major corporations
- **Time series data** spanning multiple years
- **Ground truth Q\&A pairs** for evaluation


## 🧠 AI Models \& Performance

### Question Answering System[^4]

- **Model**: RoBERTa-base fine-tuned on SQuAD 2.0
- **Features**: Confidence scoring, context chunking, overlap preservation
- **Performance**: Handles complex financial queries with professional accuracy


### Sentiment Analysis[^5]

- **Model**: FinBERT specialized for financial text
- **Features**: Document-level aggregation, temporal analysis, confidence weighting
- **Output**: Sentiment scores, distribution analysis, trend identification


### Document Retrieval[^6]

- **Technology**: FAISS indexing with sentence transformers
- **Features**: Semantic search, relevance scoring, metadata filtering
- **Performance**: Fast retrieval from large document collections


### Anomaly Detection[^7]

- **Methods**: Isolation Forest + Statistical analysis
- **Features**: Multi-metric analysis, growth rate anomalies, risk scoring
- **Applications**: Financial risk assessment, unusual pattern identification


## 🔧 Development

### Running Individual Components

Each system component can be run independently as Jupyter notebooks:

```bash
# Test individual systems
jupyter notebook qa_system.ipynb
jupyter notebook sentiment_analyzer.ipynb
jupyter notebook retrieval_system.ipynb
jupyter notebook anomaly_detector.ipynb
```


### API Development

Start the Flask development server:[^3]

```bash
python -c "exec(open('financebench_app.ipynb').read())"
```


### Testing

Run the test suite:[^8]

```bash
jupyter notebook test_api.ipynb
```
