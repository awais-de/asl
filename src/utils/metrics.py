from datasets import load_metric

def compute_bleu(predictions, references):
    """
    Compute BLEU score using sacrebleu via HuggingFace datasets.
    Args:
        predictions (list[str]): list of predicted sentences.
        references (list[str]): list of reference sentences.
    Returns:
        float: BLEU score (0-100).
    """
    bleu = load_metric("sacrebleu")
    # sacrebleu expects references as list of lists for multi-reference support
    return bleu.compute(predictions=predictions, references=[[r] for r in references])['score']

def compute_rouge(predictions, references):
    """
    Compute ROUGE scores.
    Args:
        predictions (list[str]): list of predicted sentences.
        references (list[str]): list of reference sentences.
    Returns:
        dict: ROUGE scores (rouge1, rouge2, rougeL)
    """
    rouge = load_metric("rouge")
    result = rouge.compute(predictions=predictions, references=references)
    # Convert scores to percentages and keep main ones
    return {key: value.mid.fmeasure * 100 for key, value in result.items()}

def compute_exact_match(predictions, references):
    """
    Compute exact match accuracy (case insensitive).
    Args:
        predictions (list[str]): list of predicted sentences.
        references (list[str]): list of reference sentences.
    Returns:
        float: accuracy percentage.
    """
    correct = sum(p.strip().lower() == r.strip().lower() for p, r in zip(predictions, references))
    return correct / len(references) * 100 if references else 0.0
