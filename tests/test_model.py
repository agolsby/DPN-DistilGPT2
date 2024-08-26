import unittest
import torch
from model import create_modified_distilgpt2, DPN

class TestModel(unittest.TestCase):
    def test_dpn(self):
        dpn = DPN(input_dim=100, output_dim=50)
        x = torch.randn(32, 100)
        output = dpn(x)
        self.assertEqual(output.shape, (32, 50))

    def test_modified_distilgpt2(self):
        model = create_modified_distilgpt2()
        input_ids = torch.randint(0, 1000, (32, 50))
        attention_mask = torch.ones_like(input_ids)
        output = model(input_ids, attention_mask=attention_mask)
        self.assertEqual(output[0].shape, (32, 50, model.config.vocab_size))

    def test_flash_attention(self):
        model = create_modified_distilgpt2()
        input_ids = torch.randint(0, 1000, (32, 50))
        attention_mask = torch.ones_like(input_ids)
        output = model(input_ids, attention_mask=attention_mask, output_attentions=True)
        self.assertIsNotNone(output.attentions)

if __name__ == '__main__':
    unittest.main()
