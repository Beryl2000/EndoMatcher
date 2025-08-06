import yaml
import argparse

from match_pipeline import run_feature_matching_pipeline  

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Run EndoMatcher image matching pipeline")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')

    parser.add_argument('--trained_model_path', type=str, help='Override: path to trained model')
    parser.add_argument('--sequence_root', type=str, help='Override: sequence root path')
    parser.add_argument('--out_path', type=str, help='Override: output directory')

    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.trained_model_path: config['trained_model_path'] = args.trained_model_path
    if args.sequence_root: config['sequence_root'] = args.sequence_root
    if args.out_path: config['out_path'] = args.out_path
    print(config)
    run_feature_matching_pipeline(**config)


if __name__ == '__main__':
    main()
