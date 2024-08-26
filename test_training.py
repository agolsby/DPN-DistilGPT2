import unittest
import torch
from training import DistilGPT2LightningModule, CurriculumCallback
from model import create_modified_distilgpt2
import pytorch_lightning as pl

class TestTraining(unittest.TestCase):
    def test_distilgpt2_lightning_module(self):
        model = create_modified_distilgpt2()
        lightning_model = DistilGPT2LightningModule(model)
        self.assertIsInstance(lightning_model, pl.LightningModule)

    def test_curriculum_callback(self):
        callback = CurriculumCallback(num_freezing_epochs=5, unfreeze_mlp_epoch=2)
        self.assertIsInstance(callback, pl.Callback)

    def test_training_step(self):
        model = create_modified_distilgpt2()
        lightning_model = DistilGPT2LightningModule(model)
        batch = (torch.randint(0, 1000, (4, 20)), torch.ones(4, 20))
        loss = lightning_model.training_step(batch, 0)
        self.assertIsInstance(loss, torch.Tensor)

if __name__ == '__main__':
    unittest.main()