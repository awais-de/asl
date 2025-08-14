import evaluate

def compute_bleu(predictions, references):
    """
    Compute BLEU score using sacrebleu via HuggingFace evaluate.
    Args:
        predictions (list[str]): list of predicted sentences.
        references (list[str]): list of reference sentences.
    Returns:
        float: BLEU score (0-100).
    """
    bleu = evaluate.load("sacrebleu")
    # sacrebleu expects references as list of lists for multi-reference support
    result = bleu.compute(predictions=predictions, references=[[r] for r in references])
    return result["score"]


def compute_rouge(predictions, references):
    """
    Compute ROUGE scores.
    Args:
        predictions (list[str]): list of predicted sentences.
        references (list[str]): list of reference sentences.
    Returns:
        dict: ROUGE scores (rouge1, rouge2, rougeL, rougeLsum) in percentage.
    """
    rouge = evaluate.load("rouge")
    result = rouge.compute(predictions=predictions, references=references)

    processed_scores = {}
    for key, value in result.items():
        if hasattr(value, "mid"):  # Old format
            processed_scores[key] = value.mid.fmeasure * 100
        else:  # New format (float)
            processed_scores[key] = value * 100

    return processed_scores



def compute_exact_match(predictions, references):
    """
    Compute exact match accuracy (case insensitive).
    Args:
        predictions (list[str]): list of predicted sentences.
        references (list[str]): list of reference sentences.
    Returns:
        float: accuracy percentage.
    """
    correct = sum(
        p.strip().lower() == r.strip().lower()
        for p, r in zip(predictions, references)
    )
    return (correct / len(references) * 100) if references else 0.0
