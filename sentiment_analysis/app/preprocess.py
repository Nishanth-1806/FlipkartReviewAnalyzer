import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Download necessary NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def load_data(filepath):
    """
    Load data from CSV file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)

def clean_text(text):
    """
    Clean review text:
    - Lowercase
    - Remove 'READ MORE'
    - Remove special characters
    - Remove extra whitespace
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = text.replace("read more", "")
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_dataframe(df, text_col='Review text', label_col='Ratings'):
    """
    Apply cleaning and create binary sentiment label.
    Ratings 1,2 -> Negative (0)
    Ratings 3,4,5 -> Positive (1) (Including 3 as Neutral/Positive for now, or drop it)
    """
    df = df.copy()
    df['cleaned_text'] = df[text_col].apply(clean_text)
    
    # Binary classification: 1-2 Negative, 3-5 Positive
    # Alternately: Drop 3? Let's keep it simple for now. 
    # Usually 4-5 is positive, 1-2 negative. 3 is ambiguous.
    # Let's drop 3 for clearer separation, or map 3 to positive.
    # Given the reviews, 3 often has mixed signals. 
    # Let's map 1,2 -> 0 (Negative), 3,4,5 -> 1 (Positive)
    df['sentiment'] = df[label_col].apply(lambda x: 1 if x >= 3 else 0)
    
    return df
