import re
import contractions

def expand_contractions(text: str) -> str:
    return contractions.fix(text)

def remove_punctuation(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text)

def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def clean_text(text: str) -> str:
    text = text.lower()
    text = expand_contractions(text)
    text = remove_punctuation(text)
    text = normalize_whitespace(text)
    return text
