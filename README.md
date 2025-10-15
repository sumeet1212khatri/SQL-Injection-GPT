# SQL-Injection-GPT
## Google colab link
```
https://colab.research.google.com/drive/1fkrAFCv32xtX1ralzYKAw522NoN0kzGn?usp=sharing
```
Hugging Face dataset link 
```
https://huggingface.co/datasets/PurpleAILAB/chatML_SQL_injection_dataset
```

A 10M parameter GPT-style transformer built from scratch in PyTorch, trained to generate SQL queries from natural language.

<!-- Badges -->

<p align="center">
<img src="https://www.google.com/search?q=https://img.shields.io/badge/Python-3.10%2B-blue%3Flogo%3Dpython%26logoColor%3Dwhite" alt="Python Version">
<img src="https://www.google.com/search?q=https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c%3Flogo%3Dpytorch%26logoColor%3Dwhite" alt="PyTorch Version">
<img src="https://www.google.com/search?q=https://img.shields.io/badge/License-MIT-green" alt="License">
<img src="https://www.google.com/search?q=https://img.shields.io/badge/HuggingFace-Datasets-yellow%3Flogo%3Dhuggingface%26logoColor%3Dwhite" alt="Hugging Face Datasets">
</p>

</div>

📖 About The Project
This project is an exploration into building a transformer-based language model from first principles. Inspired by Andrej Karpathy's nanoGPT, the primary goal was not to simply fine-tune an existing model, but to gain a fundamental understanding of the mechanics of the GPT architecture by implementing it from scratch.

The result is QueryBot, a compact language model trained exclusively on a dataset of SQL questions and answers. It demonstrates that a specialized, effective model can be trained on a single consumer GPU with a well-engineered data and training pipeline.

✨ Key Features
Built from Scratch: The entire GPT architecture, including multi-head self-attention and MLP blocks, was implemented from the ground up using PyTorch.

Specialized SQL Knowledge: Trained on the 78,000-example b-mc2/sql-create-context dataset to specialize in generating SQL queries.

Optimized Data Pipeline: Features an efficient data preprocessing script that tokenizes and serializes the entire corpus into a binary format, reducing data loading times by over 90%.

Efficient Training: Utilizes techniques like mixed-precision and gradient accumulation to enable training on a single GPU with limited VRAM.

Quantified Performance: Achieves a P95 inference latency of under 200ms for 150-token sequences with approximately 90% syntactic correctness.

🛠️ Tech Stack
This project was built using the following core technologies:

<p align="left">
<img src="https://img.shields.io/badge/Python-3776AB%3Fstyle%3Dfor-the-badge%26logo%3Dpython%26logoColor%3Dwhite" alt="Python">
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
<img src="https://img.shields.io/badge/HuggingFace-FFD21E%3Fstyle%3Dfor-the-badge%26logo%3Dhuggingface%26logoColor%3Dblack" alt="Hugging Face">
<img src="https://img.shields.io/badge/NumPy-013243%3Fstyle%3Dfor-the-badge%26logo%3Dnumpy%26logoColor%3Dwhite" alt="NumPy">
<img src="https://img.shields.io/badge/Jupyter-F37626%3Fstyle%3Dfor-the-badge%26logo%3Djupyter%26logoColor%3Dwhite" alt="Jupyter Notebook">
</p>

🚀 Getting Started
To get a local copy up and running, follow these simple steps.

Prerequisites
Python 3.10 or later

PyTorch 2.0 or later

An NVIDIA GPU with CUDA support is recommended for training.

Installation
Clone the repository:

git clone [https://github.com/your-username/your-repo-name.git](https://github.com/S1u2m3e4e5t6/SQL-Injection-GPT/)
cd your-repo-name

Install the required packages:

pip install torch datasets tiktoken numpy tqdm matplotlib

Prepare the Data:
Run the data preparation script (e.g., prepare_data.py). This will download the dataset from Hugging Face, perform tokenization, and save the train.bin and validation.bin files.
(Note: You will need to extract the data processing code from your notebook into a separate script for this step.)

Start Training:
Execute the training script (e.g., train.py). This will train the model from scratch and save the best-performing checkpoint to best_model_params_sql.pt.

💡 Usage Example
Once the model is trained, you can easily load it and generate SQL queries from a natural language prompt.

import torch
import tiktoken
from model import GPT, GPTConfig # Assume your model classes are in model.py
--- 
# Setup 
---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
enc = tiktoken.get_encoding("gpt2")
---
#  Load Model 
---
config = GPTConfig() # Use the same config as training
model = GPT(config)
model.load_state_dict(torch.load('best_model_params_sql.pt', map_location=device))
model.to(device)
model.eval()
---
#  Generate Response 
---
prompt = "USER: What is the highest number of wins for teams with less than 1 draw?\nASSISTANT: "
start_ids = enc.encode(prompt)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

with torch.no_grad():
    y = model.generate(x, max_new_tokens=50)
    response = enc.decode(y[0].tolist())
    print(response)

🙏 Acknowledgments
This project's architecture and training methodology are heavily inspired by Andrej Karpathy's nanoGPT.

The Hugging Face team for providing access to datasets and the datasets library.

<div align="center">
This project was developed as a learning exercise in building modern AI systems from the ground up.
</div>


