import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from sklearn.manifold import TSNE
from datasets import load_dataset
from torch.utils.data import DataLoader

def generate_visualizations(model, dataloader):
    print("Generating visualizations...")
    
    attention_weights = get_attention_weights(model, dataloader)
    plot_attention_heatmap(attention_weights)
    
    hidden_states = get_hidden_states(model, dataloader)
    plot_hidden_states_tsne(hidden_states)

def perform_analysis(model, dataloader):
    print("Performing analysis...")
    
    perplexity = calculate_perplexity(model, dataloader)
    print(f"Model perplexity: {perplexity}")
    
    top_k_accuracy = calculate_top_k_accuracy(model, dataloader)
    print(f"Top-k accuracy: {top_k_accuracy}")

    return {"perplexity": perplexity, "top_k_accuracy": top_k_accuracy}

def get_attention_weights(model, dataloader):
    model.eval()
    attention_weights = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = batch
            outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True)
            attention_weights.append(outputs.attentions[-1].mean(dim=0).cpu().numpy())
    return np.mean(attention_weights, axis=0)

def plot_attention_heatmap(attention_weights):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, cmap='viridis')
    plt.title("Average Attention Weights")
    plt.xlabel("Token Position")
    plt.ylabel("Token Position")
    plt.savefig("attention_heatmap.png")
    plt.close()

def get_hidden_states(model, dataloader):
    model.eval()
    hidden_states = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = batch
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states.append(outputs.hidden_states[-1][:, 0, :].cpu().numpy())
    return np.concatenate(hidden_states)

def plot_hidden_states_tsne(hidden_states):
    tsne = TSNE(n_components=2, random_state=42)
    hidden_states_2d = tsne.fit_transform(hidden_states)
    plt.figure(figsize=(10, 8))
    plt.scatter(hidden_states_2d[:, 0], hidden_states_2d[:, 1], alpha=0.5)
    plt.title("t-SNE visualization of hidden states")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig("hidden_states_tsne.png")
    plt.close()

def calculate_perplexity(model, dataloader):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            total_loss += outputs.loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)
    return torch.exp(torch.tensor(total_loss / total_tokens))

def calculate_top_k_accuracy(model, dataloader, k=5):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = batch
            outputs = model(input_ids[:, :-1], attention_mask=attention_mask[:, :-1])
            logits = outputs.logits[:, -1, :]
            _, top_k_preds = torch.topk(logits, k, dim=-1)
            correct += torch.sum(top_k_preds == input_ids[:, -1].unsqueeze(-1)).item()
            total += input_ids.size(0)
    return correct / total

def run_benchmarks(models):
    print("Running benchmarks...")
    
    benchmarks = [
        ("GLUE MRPC", load_dataset("glue", "mrpc", split="validation")),
        ("LAMBADA", load_dataset("lambada", split="validation")),
        ("HellaSwag", load_dataset("hellaswag", split="validation")),
    ]
    
    results = {}
    
    for model in models:
        model_results = {}
        for benchmark_name, dataset in benchmarks:
            if benchmark_name == "GLUE MRPC":
                accuracy = run_glue_mrpc(model, dataset)
            elif benchmark_name == "LAMBADA":
                accuracy = run_lambada(model, dataset)
            elif benchmark_name == "HellaSwag":
                accuracy = run_hellaswag(model, dataset)
            model_results[benchmark_name] = accuracy
        results[model.__class__.__name__] = model_results
    
    return results

def run_glue_mrpc(model, dataset):
    model.eval()
    correct = 0
    total = 0
    for batch in DataLoader(dataset, batch_size=32):
        inputs = model.tokenizer(batch['sentence1'], batch['sentence2'], return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        correct += (predictions == batch['label']).sum().item()
        total += len(batch['label'])
    return correct / total

def run_lambada(model, dataset):
    model.eval()
    correct = 0
    total = 0
    for batch in DataLoader(dataset, batch_size=32):
        inputs = model.tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
        target_word = batch['target_word']
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        correct += sum(model.tokenizer.decode(pred.item()) == target for pred, target in zip(predictions, target_word))
        total += len(target_word)
    return correct / total

def run_hellaswag(model, dataset):
    model.eval()
    correct = 0
    total = 0
    for batch in DataLoader(dataset, batch_size=32):
        context = batch['ctx']
        endings = batch['endings']
        label = batch['label']
        
        scores = []
        for ending in endings:
            inputs = model.tokenizer(context, ending, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            scores.append(outputs.logits[:, -1, :].mean().item())
        
        prediction = scores.index(max(scores))
        correct += (prediction == label).sum().item()
        total += len(label)
    return correct / total