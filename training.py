import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import signal

class DistilGPT2LightningModule(pl.LightningModule):
    def __init__(self, model, learning_rate=5e-5, weight_decay=0.01, accumulate_grad_batches=4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.accumulate_grad_batches = accumulate_grad_batches
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        scheduler = self.lr_schedulers()

        input_ids, attention_mask = batch
        outputs = self(input_ids, attention_mask=attention_mask)
        loss = outputs[0]
        self.log('train_loss', loss)

        loss = loss / self.accumulate_grad_batches
        self.manual_backward(loss)

        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            opt.step()
            opt.zero_grad()
            scheduler.step()

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask = batch
        outputs = self(input_ids, attention_mask=attention_mask)
        loss = outputs[0]
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.05
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

class KeyboardInterruptHandler:
    def __init__(self, trainer, model):
        self.trainer = trainer
        self.model = model
        self.original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        print("\nTraining paused. Do you want to adjust hyperparameters? (y/n)")
        choice = input().lower()
        if choice == 'y':
            self.adjust_hyperparameters()
        else:
            print("Resuming training...")
        
    def adjust_hyperparameters(self):
        hyperparams = {
            'learning_rate': self.trainer.optimizers[0].param_groups[0]['lr'],
            'weight_decay': self.trainer.optimizers[0].param_groups[0]['weight_decay'],
            'dropout': self.model.config.dropout,
            'attention_dropout': self.model.config.attention_dropout,
            'batch_size': self.trainer.train_dataloader.batch_size,
            'accumulate_grad_batches': self.trainer.accumulate_grad_batches
        }

        print("Current hyperparameters:")
        for name, value in hyperparams.items():
            print(f"{name}: {value}")

        for name in hyperparams:
            new_value = input(f"Enter new {name} (or press enter to keep current): ")
            if new_value:
                if name in ['learning_rate', 'weight_decay', 'dropout', 'attention_dropout']:
                    new_value = float(new_value)
                elif name in ['batch_size', 'accumulate_grad_batches']:
                    new_value = int(new_value)
                
                if name == 'learning_rate':
                    for param_group in self.trainer.optimizers[0].param_groups:
                        param_group['lr'] = new_value
                elif name == 'weight_decay':
                    for param_group in self.trainer.optimizers[0].param_groups:
                        if param_group['weight_decay'] > 0:
                            param_group['weight_decay'] = new_value
                elif name in ['dropout', 'attention_dropout']:
                    setattr(self.model.config, name, new_value)
                    self.model.train()  # Ensure dropout is applied
                elif name == 'batch_size':
                    self.trainer.train_dataloader.batch_sampler.batch_size = new_value
                    self.trainer.val_dataloader.batch_sampler.batch_size = new_value
                elif name == 'accumulate_grad_batches':
                    self.trainer.accumulate_grad_batches = new_value

                print(f"{name} updated to {new_value}")

        print("Hyperparameters updated. Resuming training...")

class CurriculumCallback(pl.Callback):
    def __init__(self, num_freezing_epochs, unfreeze_mlp_epoch):
        self.num_freezing_epochs = num_freezing_epochs
        self.unfreeze_mlp_epoch = unfreeze_mlp_epoch

    def on_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == 0:
            for name, param in pl_module.model.named_parameters():
                if 'dpn' not in name:
                    param.requires_grad = False
        
        if trainer.current_epoch == self.unfreeze_mlp_epoch:
            for name, param in pl_module.model.named_parameters():
                if 'mlp' in name:
                    param.requires_grad = True
        
        if trainer.current_epoch == self.num_freezing_epochs:
            for param in pl_module.parameters():
                param.requires_grad = True

def train_model(model, train_dataloader, val_dataloader, max_epochs=10, num_freezing_epochs=5, unfreeze_mlp_epoch=2, accumulate_grad_batches=4):
    lightning_model = DistilGPT2LightningModule(model, accumulate_grad_batches=accumulate_grad_batches)
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[CurriculumCallback(num_freezing_epochs, unfreeze_mlp_epoch)],
        gradient_clip_val=1.0,
        precision=16,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    
    keyboard_interrupt_handler = KeyboardInterruptHandler(trainer, model)
    
    trainer.fit(lightning_model, train_dataloader, val_dataloader)
    
    return lightning_model