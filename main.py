import argparse
import logging
from model import create_modified_distilgpt2
from data import load_dataset, create_dataloaders
from training import train_model
from evaluation import generate_visualizations, perform_analysis, run_benchmarks
from optimization import run_hyperparameter_optimization
from transformers import DistilGPTModel, GPT2Tokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_baseline():
    logger.info("Running baseline performance with original DistilGPT-2")
    model = DistilGPTModel.from_pretrained('distilgpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    texts = load_dataset()
    train_dataloader, val_dataloader = create_dataloaders(texts, tokenizer, batch_size=32, max_length=512, curriculum_phase=0)
    
    baseline_results = perform_analysis(model, val_dataloader)
    benchmark_results = run_benchmarks([model])
    
    logger.info("Baseline DistilGPT-2 Results:")
    logger.info(f"Analysis: {baseline_results}")
    logger.info(f"Benchmarks: {benchmark_results}")
    
    return baseline_results, benchmark_results

def main(mode, config):
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token
    texts = load_dataset()

    if mode in ['A', 'B', 'C']:
        baseline_results, baseline_benchmarks = run_baseline()
        
        if mode == 'A':
            logger.info("Running hyperparameter search...")
            best_params = run_hyperparameter_optimization(texts, tokenizer)
            model = create_modified_distilgpt2(config_kwargs=best_params)
        elif mode == 'B':
            logger.info("Fine-tuning new architecture with manual hyperparameters...")
            model = create_modified_distilgpt2()
        else:  # Mode C
            logger.info("Fine-tuning original model...")
            model = DistilGPTModel.from_pretrained('distilgpt2')

        train_dataloader, val_dataloader = create_dataloaders(texts, tokenizer, batch_size=config['batch_size'], max_length=512, curriculum_phase=0)

        trained_model = train_model(model, train_dataloader, val_dataloader, max_epochs=config['max_epochs'], accumulate_grad_batches=config['accumulate_grad_batches'])

        generate_visualizations(trained_model, val_dataloader)
        analysis_results = perform_analysis(trained_model, val_dataloader)
        benchmark_results = run_benchmarks([trained_model])

        logger.info("Training completed. Results:")
        logger.info(f"Analysis: {analysis_results}")
        logger.info(f"Benchmarks: {benchmark_results}")

    else:  # Mode D
        logger.info("Running benchmarks...")
        models_to_benchmark = [
            DistilGPTModel.from_pretrained('distilgpt2'),
            create_modified_distilgpt2(),
            # Add any other models you want to benchmark
        ]
        benchmark_results = run_benchmarks(models_to_benchmark)
        logger.info("Benchmark results:")
        logger.info(benchmark_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DistilGPT-2 Modification Project")
    parser.add_argument('mode', choices=['A', 'B', 'C', 'D'], help="Execution mode")
    args = parser.parse_args()

    config = {
        'batch_size': 32,
        'max_epochs': 10,
        'accumulate_grad_batches': 4,
    }

    main(args.mode, config)