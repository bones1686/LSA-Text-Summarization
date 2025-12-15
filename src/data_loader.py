from datasets import load_dataset
import pandas as pd

class DataLoader:
    def __init__(self, dataset_name="multi_news", split="train"):
        self.dataset_name = dataset_name
        self.split = split
        self.dataset = None

    def load_data(self, n_samples=None):
        """
        Loads the dataset.
        Args:
            n_samples: Number of samples to load (optional). If None, loads full split.
        Returns:
            pd.DataFrame: DataFrame containing 'document' and 'summary'.
        """
        print(f"Loading {self.dataset_name} dataset ({self.split} split)...")
        try:
            # Added trust_remote_code=True to fix the error
            self.dataset = load_dataset(self.dataset_name, split=self.split, trust_remote_code=True)
            
            if n_samples:
                self.dataset = self.dataset.select(range(min(n_samples, len(self.dataset))))
            
            df = pd.DataFrame(self.dataset)
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

if __name__ == "__main__":
    loader = DataLoader(split="train")
    df = loader.load_data(n_samples=5)
    print(df.head())
    print(f"Loaded {len(df)} samples.")

