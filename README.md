# DistilGPT-2 Modification Project

This project implements a modified version of the DistilGPT-2 model with nonlinear down-projection in its MLP layers. It includes features such as curriculum learning, hyperparameter optimization, and comprehensive evaluation.

## Features

- Modified DistilGPT-2 architecture with nonlinear down-projection
- Curriculum learning
- Flash Attention 2 implementation
- Hyperparameter optimization using Optuna
- Comprehensive evaluation and benchmarking
- On-the-fly hyperparameter adjustment during training

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/agolsby/DPN-DistilGPT2.git
   cd DPN-DistilGPT2
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

The project can be run in four different modes:

- Mode A: Hyperparameter search
- Mode B: Fine-tune new architecture with manual hyperparameters
- Mode C: Fine-tune original model
- Mode D: Run benchmarks

To run the project, use the following command:

```
python cli.py <mode> [--config CONFIG_FILE] [--learning_rate LR] [--batch_size BS] [--max_epochs EPOCHS] [--accumulate_grad_batches ACCUM]
```

For example:

```
python cli.py A --learning_rate 1e-4 --batch_size 32 --max_epochs 10 --accumulate_grad_batches 4
```

## Testing

To run the integration tests:

```
python -m unittest integration_test.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
