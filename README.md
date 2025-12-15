# LSA Text Summarization Pipeline

This repository contains an implementation of an extractive text summarization system based on Latent Semantic Analysis (LSA). A key feature of this project is the custom implementation of the Singular Value Decomposition (SVD) algorithm using the Lanczos method with full reorthogonalization.

The system is evaluated on the Multi-News dataset and compared against baselines (TextRank, sklearn LSA) and SOTA models (PEGASUS/BART).

## Features

- **Custom Linear Algebra Core**: Implementation of the Lanczos algorithm for SVD (`LanczosSVD`) with stable handling of eigenvalues.
- **Advanced Summarization Logic**:
  - **Rank Selection**: Optimal rank determination using the Gavish-Donoho method.
  - **Sentence Selection**: Maximal Marginal Relevance (MMR) to balance relevance and diversity in the summary.
- **Preprocessing**: Robust text cleaning and vectorization using `spaCy` and TF-IDF.
- **Evaluation**: Comprehensive benchmark including ROUGE-1, ROUGE-2, ROUGE-L, and BERTScore metrics.

## Project Structure

```
├── experiments/           # Scripts for running experiments and evaluation
│   ├── evaluate.py        # Main evaluation logic
│   ├── run_evaluation.py  # Wrapper to run evaluation
│   └── reference_pipeline.py # Baseline/Reference implementation
├── src/                   # Source code
│   ├── linear_algebra/    # Custom SVD implementation
│   ├── summarizer/        # Summarization logic (MMR, Rank Selector, LSA Class)
│   ├── data_loader.py     # Data loading utilities (Multi-News)
│   ├── preprocessor.py    # Text preprocessing with spaCy
│   └── vectorizer.py      # TF-IDF vectorization wrapper
├── tests/                 # Unit tests
│   └── test_svd.py        # Tests for custom SVD vs Scipy
└── requirements.txt       # Project dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the spaCy language model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

### Running Evaluation

To run the full evaluation pipeline comparing Custom LSA, Sklearn LSA, TextRank, and SOTA models:

```bash
python experiments/run_evaluation.py
```

This will generate a results report (e.g., CSV or console output) with ROUGE and BERTScore metrics.

### Running Tests

To verify the correctness of the custom Lanczos SVD implementation:

```bash
pytest tests/
# or
python -m unittest discover tests
```

## Methodology

1.  **Preprocessing**: Text is tokenized and lemmatized; stop words are removed.
2.  **Vectorization**: Sentences are converted to a TF-IDF matrix.
3.  **Decomposition**: The term-sentence matrix is decomposed using Custom Lanczos SVD to capture latent topics.
4.  **Selection**: Sentences are ranked based on their contribution to the top latent concepts, refined by MMR to reduce redundancy.

