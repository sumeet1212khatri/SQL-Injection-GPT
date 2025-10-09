# SQL-Injection-GPT
## Google colab link
```
https://colab.research.google.com/drive/1fkrAFCv32xtX1ralzYKAw522NoN0kzGn?usp=sharing
```
Hugging Face dataset link 
```
https://huggingface.co/datasets/PurpleAILAB/chatML_SQL_injection_dataset
```

<img width="562" height="455" alt="image" src="https://github.com/user-attachments/assets/11c68857-f4d4-47f8-b56f-b6f5ee08ba34" />




📖 About The Project
QueryBot is an educational project aimed at building a small yet powerful language model (SLM) from scratch. Inspired by Andrej Karpathy's nanoGPT, this project implements a standard GPT (Generative Pre-trained Transformer) architecture and fine-tunes it on the specific domain of SQL queries.

The result is a lightweight model that can run on a consumer GPU and provide helpful answers to a wide range of SQL-related queries.

✨ Key Features
Built from Scratch: The entire GPT architecture, including self-attention and MLP blocks, is implemented from the ground up using PyTorch.

Specialized Knowledge: Fine-tuned on the b-mc2/sql-create-context dataset to specialize in SQL.

Lightweight & Efficient: Small enough to be trained and run on a single consumer GPU.

Clear & Educational Code: The codebase is written to be easy to understand, making it an excellent resource for learning about transformers.

End-to-End Pipeline: Includes scripts for data preprocessing, tokenization, training, and inference.

🛠️ Tech Stack
This project is built using modern data science and machine learning technologies:

<p align="left">
<img src="https://files.codingninjas.in/article_images/custom-upload-1679511407.jpg" alt="Python">
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
<img src="https://www.google.com/search?q=https://img.shields.io/badge/HuggingFace-FFD21E%3Fstyle%3Dfor-the-badge%26logo%3Dhuggingface%26logoColor%3Dblack" alt="Hugging Face">
<img src="https://www.google.com/search?q=https://img.shields.io/badge/NumPy-013243%3Fstyle%3Dfor-the-badge%26logo%3Dnumpy%26logoColor%3Dwhite" alt="NumPy">
<img src="https://www.google.com/search?q=https://img.shields.io/badge/Jupyter-F37626%3Fstyle%3Dfor-the-badge%26logo%3Djupyter%26logoColor%3Dwhite" alt="Jupyter Notebook">
</p>

🚀 Getting Started
Follow these steps to get the project up and running on your local machine.

Prerequisites
Ensure you have Python 3.10 or later and pip installed.


Install the required packages:

pip install torch datasets tiktoken numpy tqdm matplotlib

Prepare the Data:
Run the data preparation script. This will download the dataset, tokenize it, and create binary files for training.
(You can integrate the Python code from Part 2 into a prepare_data.py file).

Start Training:
Run the training script to train the model from scratch. The best model will be saved as best_model_params_sql.pt.
(You can integrate the Python code from Part 4 & 5 into a train.py file).

💡 Usage Example
Once the model is trained, you can use it to generate answers to your SQL questions. Here’s a quick example:

---
#  Generate Response 
---
prompt = "USER: What are the main types of SQL injection?\nASSISTANT: "

start_ids = enc.encode(prompt)

x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

with torch.no_grad():
    y = model.generate(x, max_new_tokens=150, temperature=0.7)
    response = enc.decode(y[0].tolist())
    print(response)

## Expected Output:
### USER: What are the main types of SQL injection?
### ASSISTANT: The main types of SQL injection are in-band SQLi (the most common),
### inferential SQLi (also known as blind SQLi), and out-of-band SQLi...

📈 Training Performance
The model was trained for 5,000 iterations. The validation loss steadily decreased, indicating that the model was successfully learning the patterns in the data.

Validation Loss Curve:
(Replace this text with an image of your matplotlib graph)

🤝 Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

📜 License
Distributed under the MIT License. See the LICENSE file for more information.

🙏 Acknowledgments
This project is heavily inspired by Andrej Karpathy's amazing nanoGPT.

The Hugging Face team for their incredible datasets library.





