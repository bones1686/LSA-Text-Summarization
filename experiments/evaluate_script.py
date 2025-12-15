import sys
import os
import pandas as pd
import numpy as np
import evaluate
import spacy
import pytextrank
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader
from src.summarizer.lsa_summarizer import LSASummarizer
from src.preprocessor import TextPreprocessor

class Evaluator:
    def __init__(self, n_samples=5):
        self.n_samples = n_samples
        self.rouge = evaluate.load("rouge")
        # BERTScore requires 'bert_score' library and model download. 
        # Instantiating it might be slow, so we can skip or do it on demand.
        try:
            from bert_score import score
            self.bert_score_fn = score
        except ImportError:
            self.bert_score_fn = None
            print("BERTScore not found. Skipping.")

        # Setup Spacy with TextRank
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.nlp.add_pipe("textrank")
        except:
            print("Could not load spacy or pytextrank properly.")
            self.nlp = None

        # Setup SOTA
        try:
            print("Loading SOTA model (distilbart)...")
            self.summarizer_pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        except:
            print("Could not load transformers pipeline.")
            self.summarizer_pipeline = None

        # Custom Summarizer
        self.custom_summarizer = LSASummarizer()

    def get_baseline_summary(self, text, n_sentences=3):
        if not self.nlp:
            return text[:500]
        doc = self.nlp(text)
        summary_sents = []
        # pytextrank summary
        for sent in doc._.textrank.summary(limit_sentences=n_sentences):
            summary_sents.append(sent.text)
        return " ".join(summary_sents)

    def get_sklearn_lsa_summary(self, text, n_sentences=3):
        # Quick implementation of Sklearn LSA for comparison
        preprocessor = TextPreprocessor()
        sentences = preprocessor.preprocess_text(text)
        if not sentences:
            return ""
        corpus = [s['processed'] for s in sentences]
        originals = [s['original'] for s in sentences]
        
        if len(corpus) < n_sentences:
            return " ".join(originals)
            
        tfidf = TfidfVectorizer()
        matrix = tfidf.fit_transform(corpus)
        
        svd = TruncatedSVD(n_components=min(matrix.shape)-1)
        svd.fit(matrix)
        
        # Simple selection strategy: Top sentences from 1st component
        # (This is a naive LSA strategy often used as baseline)
        comps = svd.components_[0] # First singular vector (V^T)
        # But wait, we want sentence scores.
        # Sentences are rows. U * Sigma.
        # sklearn svd.transform returns U * Sigma
        
        matrix_reduced = svd.transform(matrix)
        # Use first component magnitude? Or sum of top k?
        # Let's use first component as dominant topic.
        scores = matrix_reduced[:, 0]
        
        top_indices = np.argsort(scores)[::-1][:n_sentences]
        top_indices = sorted(top_indices)
        
        return " ".join([originals[i] for i in top_indices])

    def get_sota_summary(self, text):
        if not self.summarizer_pipeline:
            return ""
        # Truncate text to fit model max input
        input_text = text[:1024*4] 
        try:
            res = self.summarizer_pipeline(input_text, max_length=150, min_length=40, do_sample=False)
            return res[0]['summary_text']
        except Exception as e:
            print(f"SOTA failed: {e}")
            return ""

    def evaluate(self):
        loader = DataLoader(split="test")
        df = loader.load_data(n_samples=self.n_samples)
        
        results = []
        
        for idx, row in df.iterrows():
            print(f"Processing sample {idx+1}/{self.n_samples}...")
            document = row['document']
            reference = row['summary']
            
            # 1. Baseline
            summ_baseline = self.get_baseline_summary(document)
            
            # 2. Sklearn LSA
            summ_sklearn = self.get_sklearn_lsa_summary(document)
            
            # 3. Custom LSA
            summ_custom = self.custom_summarizer.summarize(document, summary_length=3)
            
            # 4. SOTA
            summ_sota = self.get_sota_summary(document)
            
            # Calculate Scores (ROUGE-1 as representative)
            row_res = {
                'id': idx,
                'ref_len': len(reference),
                'custom_len': len(summ_custom)
            }
            
            methods = {
                'Baseline': summ_baseline,
                'Sklearn': summ_sklearn,
                'Custom': summ_custom,
                'SOTA': summ_sota
            }
            
            for name, pred in methods.items():
                if not pred:
                    row_res[f'{name}_R1'] = 0.0
                    continue
                    
                rouge_res = self.rouge.compute(predictions=[pred], references=[reference])
                # rouge_res['rouge1'] is a Score object with mid, low, high. Use mid.fmeasure
                row_res[f'{name}_R1'] = rouge_res['rouge1'].mid.fmeasure
                
                # BERTScore is slow, maybe skip for single run or do it if requested
                # if self.bert_score_fn:
                #    P, R, F1 = self.bert_score_fn([pred], [reference], lang="en", verbose=False)
                #    row_res[f'{name}_BERT'] = F1.mean().item()
            
            results.append(row_res)
            
        results_df = pd.DataFrame(results)
        print("\nResults Summary (ROUGE-1 F1):")
        print(results_df.mean(numeric_only=True))
        
        results_df.to_csv('evaluation_results.csv', index=False)
        print("Results saved to evaluation_results.csv")

if __name__ == "__main__":
    # Ensure resources are downloaded
    # import nltk
    # nltk.download('punkt')
    
    evaluator = Evaluator(n_samples=3) # Small sample for demo
    evaluator.evaluate()

