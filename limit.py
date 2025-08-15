from pathlib import Path
import json
import jsonlines

from src.utils.artifact_names import (
    ARTIFACTS_FOLDER,
    ASLG_PC12_CLEAN_JSONL,
    ASLG_PC12_CLEAN_LIMIT_JSONL,
    GLOSS_TO_VIDEOID_MAP_JSON,
)

aslg_pc12_cleaned_limit_path = Path(f"{ARTIFACTS_FOLDER}/{ASLG_PC12_CLEAN_LIMIT_JSONL}")
gloss_video_map_path        = Path(f"{ARTIFACTS_FOLDER}/{GLOSS_TO_VIDEOID_MAP_JSON}")
aslg_pc12_cleaned_path      = Path(f"{ARTIFACTS_FOLDER}/{ASLG_PC12_CLEAN_JSONL}")

# Load gloss->videos and normalize keys
with open(gloss_video_map_path, "r", encoding="utf-8") as f:
    gloss_to_vids = json.load(f)
available_glosses = {str(k).strip().lower() for k in gloss_to_vids.keys()}
print(f"Number of glosses with videos: {len(available_glosses)}")

total_samples = 0
kept_samples  = 0
threshold     = 0.6  # 60% token match required

# Ensure output directory exists
aslg_pc12_cleaned_limit_path.parent.mkdir(parents=True, exist_ok=True)

def tokenize_gloss(g):
    """Normalize gloss into lowercase tokens."""
    if isinstance(g, str):
        return [t.strip().lower() for t in g.split() if t.strip()]
    if isinstance(g, list):
        return [str(t).strip().lower() for t in g if str(t).strip()]
    return []

# Stream filter & write
with jsonlines.open(aslg_pc12_cleaned_path, mode="r") as reader, \
     jsonlines.open(aslg_pc12_cleaned_limit_path, mode="w") as writer:
    for i, item in enumerate(reader):
        total_samples += 1
        tokens = tokenize_gloss(item.get("gloss", ""))

        if tokens:
            matches = sum(1 for tok in tokens if tok in available_glosses)
            ratio   = matches / len(tokens)

            # Keep only if ≥ 60% of tokens match
            if ratio >= threshold:
                writer.write(item)
                kept_samples += 1

print(f"Total samples in original dataset: {total_samples}")
print(f"Samples kept after filtering (≥{int(threshold*100)}% token match): {kept_samples} "
      f"({kept_samples/total_samples:.2%} retained)")
print(f"Filtered dataset saved to: {aslg_pc12_cleaned_limit_path}")
