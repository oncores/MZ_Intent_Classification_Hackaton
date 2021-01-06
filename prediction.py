import argparse
import pandas as pd
import run_seq_cls
from src import mlp_classifier
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def main(main_args):
    main_args.config_dir = 'config'
    main_args.config_file = 'koelectra-base-v3.json'

    main_args.task='mzitc'
    main_args.output_dir="koelectra-base-v3-mzitc-ckpt"
    all_logits = run_seq_cls.main(main_args)

    main_args.task='mzhitc'
    main_args.output_dir="koelectra-base-v3-mzhitc-ckpt"
    second_logits = run_seq_cls.main(main_args)

    main_args.task='mzhitc_third'
    main_args.output_dir="koelectra-base-v3-mzhitc-third-ckpt"
    third_logits = run_seq_cls.main(main_args)

    result = mlp_classifier.main(all_logits, second_logits, third_logits)
    pd.DataFrame(result).to_csv('./output/' + main_args.output_text, header=None, index=False)

    return None

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

    # cli_parser.add_argument("--task", type=str, required=False)
    # cli_parser.add_argument("--config_dir", type=str, default="config")
    # cli_parser.add_argument("--config_file", type=str, required=False)
    cli_parser.add_argument("--input_text", type=str, required=True)
    cli_parser.add_argument("--output_text", type=str, required=True)
    #
    cli_args = cli_parser.parse_args()

    main(cli_args)
