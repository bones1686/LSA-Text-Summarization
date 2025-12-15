import spacy
import re

class TextPreprocessor:
    def __init__(self, model_name="en_core_web_sm", min_sentence_length=10):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model '{model_name}' not found. Downloading...")
            from spacy.cli import download
            download(model_name)
            self.nlp = spacy.load(model_name)
            
        self.min_sentence_length = min_sentence_length

    def preprocess_text(self, text):
        """
        Splits text into sentences, cleans and lemmatizes them.
        Args:
            text (str): Input text.
        Returns:
            list: List of sentence objects (dict with original and processed text).
        """
        if not isinstance(text, str):
            return []

        # Cleanup some common artifacts in multi-news
        text = text.replace('NEWLINE_CHAR', ' ')
        text = re.sub(r'\s+', ' ', text).strip()

        doc = self.nlp(text)
        sentences = []

        for sent in doc.sents:
            original_text = sent.text.strip()
            if len(original_text) < self.min_sentence_length:
                continue

            # Lemmatization and filtering stopwords/punctuation
            tokens = [
                token.lemma_.lower() 
                for token in sent 
                if not token.is_stop and not token.is_punct and not token.is_space
            ]
            
            if len(tokens) < 3: # Filter very short sentences after processing
                continue

            processed_text = " ".join(tokens)
            
            sentences.append({
                'original': original_text,
                'processed': processed_text,
                'tokens': tokens
            })

        return sentences

    def batch_preprocess(self, texts):
        return [self.preprocess_text(text) for text in texts]

