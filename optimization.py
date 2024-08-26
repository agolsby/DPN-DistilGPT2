import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
from model import create_modified_distilgpt2
from training import DistilGPT2LightningModule, CurriculumCallback
from data import create_dataloaders

def objective(trial, texts, tokenizer):
    # Suggest values of the hyperparameters using a trial object.
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    # Create model with trial hyperparameters
    model = create_modified_distilgpt2(config_kwargs={"dropout": dropout})
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(texts, tokenizer, batch_size=batch_size, max_length=512, curriculum_phase=0)
    
    # Create Lightning module
    lightning_model = DistilGPT2LightningModule(model, learning_rate=learning_rate, weight_decay=weight_decay)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=10,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[
            CurriculumCallback(num_freezing_epochs=5, unfreeze_mlp_epoch=2),
            PyTorchLightningPruningCallback(trial, monitor="val_loss")
        ],
        gradient_clip_val=1.0,
    )
    
    # Train the model
    trainer.fit(lightning_model, train_dataloader, val_dataloader)
    
    return trainer.callback_metrics["val_loss"].item()

def run_hyperparameter_optimization(texts, tokenizer, n_trials=100):
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, texts, tokenizer), n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return trial.params