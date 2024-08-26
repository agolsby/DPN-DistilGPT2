import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from sklearn.model_selection import train_test_split
import datasets

class CurriculumDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length, curriculum_phase):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.curriculum_phase = curriculum_phase

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.curriculum_phase == 0:
            max_len = min(50, self.max_length)
        elif self.curriculum_phase == 1:
            max_len = min(100, self.max_length)
        elif self.curriculum_phase == 2:
            max_len = min(200, self.max_length)
        else:
            max_len = self.max_length

        encoded = self.tokenizer.encode_plus(
            self.texts[idx],
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return encoded['input_ids'].squeeze(), encoded['attention_mask'].squeeze()

def create_dataloaders(texts, tokenizer, batch_size, max_length, curriculum_phase, num_workers=4):
    train_texts, val_texts = train_test_split(texts, test_size=0.1, random_state=42)

    train_dataset = CurriculumDataset(train_texts, tokenizer, max_length, curriculum_phase)
    val_dataset = CurriculumDataset(val_texts, tokenizer, max_length, curriculum_phase)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )

    return train_dataloader, val_dataloader

def load_dataset(dataset_name="wikitext", subset="wikitext-103-raw-v1", split="train"):
    dataset = datasets.load_dataset(dataset_name, subset, split=split)
    texts = dataset["text"]
    # Filter out empty strings and very short texts
    texts = [text for text in texts if len(text.strip()) > 50]
    return texts