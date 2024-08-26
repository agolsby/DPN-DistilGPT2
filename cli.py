import argparse
import json
from main import main as run_main

def parse_args():
    parser = argparse.ArgumentParser(description="DistilGPT-2 Modification Project")
    parser.add_argument('mode', choices=['A', 'B', 'C', 'D'], help="Execution mode")
    parser.add_argument('--config', type=str, help="Path to configuration file")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--max_epochs', type=int, default=10, help="Maximum number of epochs")
    parser.add_argument('--accumulate_grad_batches', type=int, default=4, help="Number of batches to accumulate gradients")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    args = parse_args()
    
    if args.config:
        config = load_config(args.config)
    else:
        config = {
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'max_epochs': args.max_epochs,
            'accumulate_grad_batches': args.accumulate_grad_batches
        }
    
    run_main(args.mode, config)

if __name__ == "__main__":
    main()