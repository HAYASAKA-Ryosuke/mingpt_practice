import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from model import GPT, GPT1Config

class SimpleDataset(Dataset):
    def __init__(self, texts, vocab, block_size):
        self.vocab = vocab
        self.block_size = block_size
        self.data = [self.encode(text) for text in texts]

    def encode(self, text):
        return [self.vocab.get(token, 0) for token in text.split()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        x = torch.tensor(d[:-1], dtype=torch.long)
        y = torch.tensor(d[1:], dtype=torch.long)
        return x, y
    
EPOCHS = 100
texts = ["This is a pen.", "This is a ramen.", "This is an apple.", "I am a man."]
vocab = {word: i for i, word in enumerate(set(" ".join(texts).split()))}
block_size = max(len(text.split()) for text in texts) + 1  # 最大のトークン数をもとにブロックサイズを設定

dataset = SimpleDataset(texts, vocab, block_size=block_size)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

config = GPT1Config(vocab_size=len(vocab) + 1, block_size=block_size)  # 語彙サイズに1を足して未知トークンを考慮
model = GPT(config)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    for idx, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        _, loss = model(x, targets=y)
        loss.backward()
        optimizer.step()
        if idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {idx}, Loss {loss.item()}")

def predict_next_word(model, text, vocab):
    tokens = [vocab.get(token, 0) for token in text.split()]
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_token_id = torch.argmax(probs, dim=-1).item()
    return next_token_id

# "This"を投げて推論
next_word_id = predict_next_word(model, "This", vocab)
inverse_vocab = {v: k for k, v in vocab.items()}
predicted_word = inverse_vocab[next_word_id]

print(f'次のトークン: {predicted_word}')
