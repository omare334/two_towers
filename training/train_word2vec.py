import pickle

import torch
import torch.optim as optim
import wandb
from tqdm import tqdm

from model.w2v_model import Word2Vec

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading corpus")
with open("dataset/processed.pkl", "rb") as f:
    (
        corpus,
        tokens,  # corpus as tokens
        words_to_ids,
        ids_to_words,
    ) = pickle.load(f)
print("Loaded corpus")


print("Init model...")
embed_dim = 50
model = Word2Vec(embed_dim, len(words_to_ids)).to(device)

old_model = Word2Vec(50,50000)
new_model = Word2Vec(50,60000)

new_model.center_embed.weight[:50000] = old_model.center_embed.weights

model_path = "checkpoints/w2v_epoch_15.pth"
model.load_state_dict(torch.load(model_path, weights_only=True))

print("Model initialised")

optimizer = optim.Adam(model.parameters(), lr=0.015)

context_window = 2
batch_size = 500_000

print("Loading dataset")
load_path = "dataset/processed/text8_set.pth"
input_tensor, target_tensor, negs_tensor = torch.load(load_path)

dataset = torch.utils.data.TensorDataset(input_tensor, target_tensor, negs_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
print("Loaded dataset")


wandb.init(
    project="word2vec",
    name="continue lr0.015",
    config={
        "batch_size": batch_size,
        "context_window": context_window,
        "embed_dims": embed_dim,
    },
)
for epoch in range(500):
    prgs = tqdm(dataloader, desc=f"Epoch {epoch+1}")

    for inputs, targets, negs in prgs:
        inputs, targets, negs = inputs.to(device), targets.to(device), negs.to(device)
        optimizer.zero_grad()

        loss = model.get_loss(inputs, targets, negs)

        loss.backward()

        optimizer.step()
        wandb.log({"loss": loss.item()})

    if not (epoch + 1) % 5:
        save_path = f"checkpoints/w2v_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
wandb.finish()
