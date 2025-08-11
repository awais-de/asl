import spacy
import contractions

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Expand contractions in the whole text
    expanded_text = contractions.fix(text)
    
    # Process expanded text with spaCy
    doc = nlp(expanded_text)
    
    # Prepare tokens: lowercase, alphabetic only (adjust as needed)
    tokens = [token.text.lower() for token in doc if not token.is_space and token.text.isalpha()]
    
    return doc, tokens

if __name__ == "__main__":
    text = "I'm excited to learn sign language. Barack Obama was the 44th president of the USA."

    doc, tokens = preprocess_text(text)
    
    print("Final normalized tokens:")
    print(tokens)
    print("\nDetailed token info:")
    print("Token\tLemma\tPOS Tag\tDependency\tHead Token")
    for token in doc:
        print(f"{token.text}\t{token.lemma_}\t{token.pos_}\t{token.dep_}\t{token.head.text}")
    
    print("\nNamed Entities:")
    for ent in doc.ents:
        print(f"Text: {ent.text}, Label: {ent.label_}")
