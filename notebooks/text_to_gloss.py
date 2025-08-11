import nltk
import spacy
from nltk.corpus import wordnet as wn

# Download needed resources
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load English model for POS tagging
nlp = spacy.load("en_core_web_sm")

# POS mapping from spaCy to WordNet
POS_MAP = {
    "NOUN": wn.NOUN,
    "VERB": wn.VERB,
    "ADJ": wn.ADJ,
    "ADV": wn.ADV
}

def load_gloss_vocab_from_txt(filepath):
    gloss_vocab = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                gloss_word = parts[1].upper()
                gloss_vocab.append(gloss_word)
    return gloss_vocab

def find_synonym_in_vocab(word, pos_tag, gloss_vocab):
    """
    Find a synonym for `word` that exists in gloss_vocab, filtered by POS.
    """
    word = word.lower()
    gloss_set = {g.lower() for g in gloss_vocab}
    wn_pos = POS_MAP.get(pos_tag, None)

    if wn_pos:
        synsets = wn.synsets(word, pos=wn_pos)
    else:
        synsets = wn.synsets(word)  # Fallback if POS not in mapping

    for syn in synsets:
        for lemma in syn.lemmas():
            lemma_word = lemma.name().lower().replace("_", " ")
            if lemma_word in gloss_set:
                return lemma_word.upper()
    return None

def text_to_gloss(tokens, gloss_vocab, strategy="fallback"):
    """
    Convert tokens to gloss sequence using:
      1. Direct match
      2. POS-aware synonym match
      3. OOV strategy
    """
    gloss_map = {g.lower(): g for g in gloss_vocab}
    gloss_sequence = []

    # Process tokens with spaCy for POS tagging
    doc = nlp(" ".join(tokens))

    for token in doc:
        tok_lower = token.text.lower()
        pos_tag = token.pos_

        # Step 1: Direct match
        if tok_lower in gloss_map:
            gloss_sequence.append(gloss_map[tok_lower])
            continue

        # Step 2: POS-aware synonym match
        synonym_gloss = find_synonym_in_vocab(tok_lower, pos_tag, gloss_vocab)
        if synonym_gloss:
            gloss_sequence.append(synonym_gloss)
            continue

        # Step 3: OOV handling
        if strategy == "fallback":
            gloss_sequence.append(token.text.upper())
        elif strategy == "skip":
            continue
        elif strategy == "unknown":
            gloss_sequence.append("UNKNOWN")
        elif strategy == "fingerspell":
            gloss_sequence.extend(list(token.text.upper()))

    return gloss_sequence

# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    vocab_path = "data/raw/WLASL2000/wlasl-complete/wlasl_class_list.txt"
    gloss_vocab = load_gloss_vocab_from_txt(vocab_path)

    tokens = ["hello", "beautiful", "book", "assist", "run", "language"]

    glosses = text_to_gloss(tokens, gloss_vocab, strategy="fallback")

    print("Gloss Sequence:", glosses)
