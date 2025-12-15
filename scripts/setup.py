import subprocess
import sys

def install_requirements():
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def download_spacy_model():
    print("Downloading spaCy model en_core_web_sm...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

def download_nltk_data():
    print("Downloading NLTK data...")
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')

if __name__ == "__main__":
    try:
        install_requirements()
        download_spacy_model()
        download_nltk_data()
        print("Setup completed successfully.")
    except Exception as e:
        print(f"Setup failed: {e}")

