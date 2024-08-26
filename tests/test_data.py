import unittest
from data import CurriculumDataset, create_dataloaders, load_dataset
from transformers import GPT2Tokenizer

class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.texts = ["This is a test sentence.", "Another test sentence."]

    def test_curriculum_dataset(self):
        dataset = CurriculumDataset(self.texts, self.tokenizer, max_length=20, curriculum_phase=0)
        self.assertEqual(len(dataset), 2)
        item = dataset[0]
        self.assertIsInstance(item[0], torch.Tensor)
        self.assertIsInstance(item[1], torch.Tensor)

    def test_create_dataloaders(self):
        train_dataloader, val_dataloader = create_dataloaders(self.texts, self.tokenizer, batch_size=2, max_length=20, curriculum_phase=0)
        self.assertIsNotNone(train_dataloader)
        self.assertIsNotNone(val_dataloader)

    def test_load_dataset(self):
        texts = load_dataset(split='train[:10]')
        self.assertIsNotNone(texts)
        self.assertGreater(len(texts), 0)

if __name__ == '__main__':
    unittest.main()
