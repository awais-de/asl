import argparse
import sys

def run_load_videos():
    from src.data_loading import load_videos
    load_videos.main()

def run_preprocess():
    from src.preprocessing import preprocess
    preprocess.main()

def run_tokenize():
    from src.preprocessing.tokenize_aslg_pc12 import tokenize_aslg_pc12
    tokenize_aslg_pc12.main()

def run_train(script_args):
    from src.models import train
    train.main(script_args)

def run_evaluate(script_args):
    from src.models import evaluate
    evaluate.main(script_args)  # pass list of args here

def main():
    parser = argparse.ArgumentParser(description="Run project scripts")
    parser.add_argument('script', choices=['load_videos', 'preprocess', 'tokenize', 'train', 'evaluate'],
                        help="Which script to run")
    args, unknown = parser.parse_known_args()

    if args.script == 'load_videos':
        run_load_videos()
    elif args.script == 'preprocess':
        run_preprocess()
    elif args.script == 'tokenize':
        run_tokenize()
    elif args.script == 'train':
        run_train(unknown)
    elif args.script == 'evaluate':
        run_evaluate(unknown)
    else:
        print(f"Unknown script {args.script}")
        sys.exit(1)

if __name__ == '__main__':
    main()
