import unittest
from main import main
from model import create_modified_distilgpt2
from data import load_dataset, create_dataloaders
from training import train_model
from evaluation import generate_visualizations, perform_analysis, run_benchmarks
from transformers import GPT2Tokenizer

class IntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.texts = load_dataset(split='train[:100]')  # Use a small subset for testing
        cls.config = {'learning_rate': 5e-5, 'batch_size': 4, 'max_epochs': 2, 'accumulate_grad_batches': 2}

    def test_full_pipeline(self):
        # Test mode A (hyperparameter search)
        main('A', self.config)
        
        # Test mode B (fine-tune new architecture)
        main('B', self.config)
        
        # Test mode C (fine-tune original model)
        main('C', self.config)
        
        # Test mode D (run benchmarks)
        main('D', self.config)

    def test_model_creation(self):
        model = create_modified_distilgpt2(use_jit=True)
        self.assertIsNotNone(model)
    
    def test_data_loading(self):
        texts = load_dataset(split='train[:10]')
        self.assertIsNotNone(texts)
        self.assertGreater(len(texts), 0)
    
    def test_training(self):
        model = create_modified_distilgpt2()
        train_dataloader, val_dataloader = create_dataloaders(self.texts, self.tokenizer, batch_size=4, max_length=128, curriculum_phase=0)
        trained_model = train_model(model, train_dataloader, val_dataloader, max_epochs=1)
        self.assertIsNotNone(trained_model)
    
    def test_evaluation(self):
        model = create_modified_distilgpt2()
        _, val_dataloader = create_dataloaders(self.texts, self.tokenizer, batch_size=4, max_length=128, curriculum_phase=0)
        
        generate_visualizations(model, val_dataloader)
        analysis_results = perform_analysis(model, val_dataloader)
        benchmark_results = run_benchmarks([model])
        
        self.assertIsNotNone(analysis_results)
        self.assertIsNotNone(benchmark_results)

if __name__ == '__main__':
    unittest.main()