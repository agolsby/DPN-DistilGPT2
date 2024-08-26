import torch
import torch.nn as nn
from transformers import DistilGPTModel, DistilGPTConfig
from typing import Optional, Tuple
from flash_attn import flash_attn_func, flash_attn_kvpacked_func

class DPN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, activation='relu', dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or (input_dim + output_dim) // 2

        self.layer1 = nn.Linear(input_dim, self.hidden_dim)
        self.layer2 = nn.Linear(self.hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

class FlashAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.n_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        kv = torch.stack([k, v], dim=2)
        
        attn_output = flash_attn_kvpacked_func(q, kv, causal=True)
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.hidden_size)
        
        return self.out_proj(attn_output)

class ModifiedDistilGPT2(DistilGPTModel):
    def __init__(self, config: DistilGPTConfig):
        super().__init__(config)
        
        for layer in self.transformer.h:
            layer.attention = FlashAttention(config)
        
        for layer in self.transformer.h:
            layer.mlp.dpn = DPN(
                input_dim=config.n_embd * 4,
                output_dim=config.n_embd,
                hidden_dim=config.n_embd * 2,
                activation='gelu',
                dropout=config.dropout
            )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple:
        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        all_hidden_states = ()
        for layer in self.transformer.h:
            hidden_states = layer.attention(hidden_states, attention_mask)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = layer.mlp.dpn(hidden_states)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if output_hidden_states:
            outputs = outputs[:1] + (all_hidden_states,) + outputs[2:]

        return outputs

def create_modified_distilgpt2(pretrained_model_name: str = "distilgpt2", config_kwargs: dict = None):
    config = DistilGPTConfig.from_pretrained(pretrained_model_name, **config_kwargs) if config_kwargs else DistilGPTConfig.from_pretrained(pretrained_model_name)
    model = ModifiedDistilGPT2(config)
    
    original_model = DistilGPTModel.from_pretrained(pretrained_model_name)
    model.load_state_dict(original_model.state_dict(), strict=False)
    
    return model