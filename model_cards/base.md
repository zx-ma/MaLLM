---
license: apache-2.0
language: en
datasets:
  - roneneldan/TinyStories
---

# mallm-base

an 84-million-parameter language model trained from scratch on TinyStories for text completion.

- 16 layers, hidden size 640, 10 attention heads, RoPE, RMSNorm, GELU MLP, tied embeddings
- 8k-token byte-level BPE vocabulary, 256-token context
- 1.63 billion training tokens (3 epochs on TinyStories V2), validation loss 1.09

## usage

```python
import json, sys
from huggingface_hub import snapshot_download
from safetensors.torch import load_model
from tokenizers import Tokenizer

path = snapshot_download("zhexMa/mallm-base")
sys.path.insert(0, path)
from model import Mallm, MallmConfig

model = Mallm(MallmConfig(**json.load(open(f"{path}/config.json"))))
load_model(model, f"{path}/model.safetensors")
tok = Tokenizer.from_file(f"{path}/tokenizer.json")
```

for generation, see `generate.py` in the [code repository](https://github.com/zx-ma/MaLLM).

## limitations

best suited to english children's stories. not instruction-tuned.
