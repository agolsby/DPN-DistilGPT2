import unittest
import torch
from evaluation import generate_visualizations, perform_analysis, run_benchmarks
from model import create_modified_distilgpt2
from data import create_dataloaders
from transformers import GPT2Tokenizer

class TestEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = create_modified_distilgpt2()
        cls.tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.texts = ["This is a test sentence.", "Another test sentence."]
        _, cls.val_dataloader = create_dataloaders(cls.texts, cls.tokenizer, batch_size=2, max_length=20, curriculum_phase=0)

    def test_generate_visualizations(self):
        generate_visualizations(self.model, self.val_dataloader)
        # Check if visualization files are created
        self.assertTrue(os.path.exists("attention_heatmap.png"))
        self.assertTrue(os.path.exists("hidden_states_tsne.png"))

    def test_perform_analysis(self):
        results = perform_analysis(self.model, self.val_dataloader)
        self.assertIn("perplexity", results)
        self.assertIn("top_k_accuracy", results)

    def test_run_benchmarks(self):
        results = run_benchmarks([self.model])
        self.assertIsInstance(results, dict)
        self.assertIn(self.model.__class__.__name__, results)

if __name__ == '__main__':
    unittest.main()