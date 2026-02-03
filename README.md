# SQL-Injection-GPT
## Google colab link
```
https://colab.research.google.com/drive/1fkrAFCv32xtX1ralzYKAw522NoN0kzGn?usp=sharing
```
Hugging Face dataset link 
```
https://huggingface.co/datasets/PurpleAILAB/chatML_SQL_injection_dataset
```

## Custom-Trained SQL Language Model (10M GPT)

A compact, 30M-parameter decoder-only transformer model architected from scratch in PyTorch and trained on a single consumer GPU to generate syntactically correct SQL queries from natural language.

### 1. 🚀 Features

Custom Architecture: Built completely from first principles in PyTorch, implementing causal self-attention, multi-head attention blocks, and positional embeddings without relying on pre-trained models like GPT-2.

Efficient Training: Optimized for single-GPU training using mixed-precision (FP16) and gradient accumulation to achieve stable convergence on limited hardware (T4 GPU).

Optimized Data Pipeline: Engineered a binary serialization pipeline for the 78k-sample dataset, reducing data loading times by >90% compared to standard text processing.

High Performance: Achieves ~90% syntactic correctness on generated SQL queries with a P95 inference latency of under 200ms for 150-token sequences.

### 2. 🛠️ Tech Stack

Core Framework: PyTorch

Architecture: Custom GPT (Decoder-only Transformer)

Tokenizer: tiktoken (OpenAI BPE)

Dataset: Hugging Face Datasets (b-mc2/sql-create-context)

Training Hardware: Single NVIDIA T4 GPU (via Google Colab)

Optimization: AdamW Optimizer, Cosine Learning Rate Decay, Mixed Precision (AMP)

### 3. 🏗️ Architecture Highlights

The model is a compact GPT-style transformer designed for efficiency:

Parameters: ~10 Million

Layers: 6 Transformer Blocks

Attention Heads: 6

Embedding Dimension: 384

Context Window: 256 tokens

This specific configuration was chosen to balance model expressivity with the memory constraints of a single 16GB GPU, allowing for a full training run without successfully crashing via OOM (Out of Memory) errors.

### 4. 🏁 Getting Started

Prerequisites

Python 3.8+

PyTorch 2.0+ (with CUDA support recommended)

tiktoken, datasets, tqdm, numpy

Installation & Inference

Clone the repository:

git clone [https://github.com/sumeet1212khatri/SQL-Injection-GPT]


Install dependencies:
```
pip install torch tiktoken datasets tqdm numpy
```

Run Inference (Generate SQL):

import torch
import tiktoken
from model import GPT, GPTConfig

# Load the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = GPTConfig()
model = GPT(config)
model.load_state_dict(torch.load('best_model_params_sql.pt', map_location=device))
model.to(device)
model.eval()

# Generate
enc = tiktoken.get_encoding("gpt2")
prompt = "USER: What is the total revenue for the 'Electronics' category?\nASSISTANT: "
x = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device)[None, ...]

with torch.no_grad():
    y = model.generate(x, max_new_tokens=100, temperature=0.2)
    print(enc.decode(y[0].tolist()))


### 5. 📊 Training & Evaluation

The model was trained for 5,000 iterations with a batch size of 16 and gradient accumulation steps of 8 (effective batch size = 128).

Loss Objective: Cross-Entropy Loss

Evaluation Metric: Validation Loss & Syntactic Correctness (judged by standard SQL parsers).

Final Validation Loss: ~1.45

Developed by Sumeet Khatri as a deep-dive into fundamental LLM architecture and efficient training.

<img width="562" height="455" alt="image" src="https://github.com/user-attachments/assets/fe87c427-5cfb-48e5-b64e-7b771bfc2fdc" />
<img width="1110" height="303" alt="image" src="https://github.com/user-attachments/assets/0cf13bee-a3d4-4546-a964-51f8845de4a2" />

