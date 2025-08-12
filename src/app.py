import argparse
from src.preprocessing.text_preprocessing import preprocess_text  # your tokenization + normalization
from src.models.text_to_gloss_model import TextToGlossModel  # your basic gloss generation model
from src.data_loading.load_videos_dataset import download_and_prepare_datasets  # hypothetical combined loader
from src.preprocessing.preprocess_videos import load_wlasl_data  # loading gloss-video mappings


def main(args):
    if args.action == "download_data":
        print("Downloading datasets...")
        download_and_prepare_datasets()  # you can implement this in load_videos_dataset.py
        print("Datasets downloaded and copied.")

    elif args.action == "preprocess_text":
        raw_text = args.text
        print(f"Original text: {raw_text}")

        tokens = preprocess_text(raw_text)
        print(f"Preprocessed tokens: {tokens}")

    elif args.action == "generate_gloss":
        raw_text = args.text
        print(f"Input text: {raw_text}")

        tokens = preprocess_text(raw_text)
        model = TextToGlossModel()
        gloss_sequence = model.text_to_gloss(tokens)

        print(f"Generated gloss sequence: {gloss_sequence}")

    else:
        print("Invalid action. Choose from: download_data, preprocess_text, generate_gloss.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASL Translation System CLI")
    parser.add_argument(
        "action",
        type=str,
        choices=["download_data", "preprocess_text", "generate_gloss"],
        help="Action to perform",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="",
        help="Input text for preprocessing or gloss generation",
    )
    args = parser.parse_args()

    main(args)
