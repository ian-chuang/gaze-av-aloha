import torch
from torch import nn
from transformers import DistilBertTokenizer, DistilBertModel
from collections import OrderedDict

class CachedDistilBertEmbedder(nn.Module):
    def __init__(self, max_cache_size=1000):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased').eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.cache = OrderedDict()
        self.max_cache_size = max_cache_size

    def forward(self, texts):
        """
        Accepts a list of strings and returns a tensor of their embeddings.
        Caches results for previously seen strings.
        Batched: Only calls DistilBERT once per set of uncached texts.
        """
        # Prepare to batch-embed uncached texts
        uncached_texts = []
        uncached_indices = []
        embeddings = [None] * len(texts)
        device = next(self.model.parameters()).device

        # First, fill in cached values and collect uncached inputs
        for idx, text in enumerate(texts):
            if text in self.cache:
                emb = self.cache[text]
                self.cache.move_to_end(text)
                embeddings[idx] = emb
            else:
                uncached_texts.append(text)
                uncached_indices.append(idx)

        # Batch process uncached texts, if any
        if uncached_texts:
            inputs = self.tokenizer(
                uncached_texts,
                return_tensors="pt",
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                # [batch_size, sequence_length, hidden_size]
                batch_embs = outputs.last_hidden_state[:, 0, :]  # [CLS] token from each

            for idx, emb, text in zip(uncached_indices, batch_embs, uncached_texts):
                embeddings[idx] = emb
                self.cache[text] = emb
                if len(self.cache) > self.max_cache_size:
                    self.cache.popitem(last=False)  # LRU

        return torch.stack(embeddings)
    
    @property
    def embed_dim(self) -> int:
        """Output dimension of the model."""
        return 768
