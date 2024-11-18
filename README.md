# two_towers


To get started with environment:
```
poetry install
```

If poetry is not installed and env consistency is not a priority
```
pip install -r gpu_requirements.txt
```

If do not have `poetry` installed -> https://python-poetry.org/

## Model architecture 
This is a simple two tower model that will be imporved overtime but can get the job done
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence  # For padding sequences
from tqdm import tqdm  # For progress bar
import wandb  # Import Weights and Biases


class TowerOneRNN(nn.Module):
    def __init__(self):
        super(TowerOneRNN, self).__init__()
        self.rnn = nn.RNN(input_size=64, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 3)

    def forward(self, x, lengths):
        # Pack the padded sequences
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # Pass input through the RNN layer
        x, _ = self.rnn(x)
        
        # Unpack the output and get the last time step
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x[torch.arange(x.size(0)), lengths - 1]  # Get the last valid time step
        
        # Pass through the fully connected layer
        x = self.fc(x)
        
        return x

class TowerTwoRNN(nn.Module):
    def __init__(self):
        super(TowerTwoRNN, self).__init__()
        self.rnn = nn.RNN(input_size=64, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 3)

    def forward(self, x, lengths):
        # Pack the padded sequences
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # Pass input through the RNN layer
        x, _ = self.rnn(x)
        
        # Unpack the output and get the last time step
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x[torch.arange(x.size(0)), lengths - 1]  # Get the last valid time step
        
        # Pass through the fully connected layer
        x = self.fc(x)
        
        return x
```
## Training loop per query 
This training loop for the two-tower model is training the model in query batches.
``` python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence  # For padding sequences
from tqdm import tqdm  # For progress bar
import wandb  # Import Weights and Biases

# Initialize W&B project
wandb.init(project="twotower_training", entity="omareweis123", name='fine_tuning_skipgram/packed_embedding')

# Load the saved SkipGram model
model_save_path = "finetuned_skipgram_model.pth"
checkpoint = torch.load(model_save_path)

# Initialize the SkipGram model
skipgram_model = SkipGramFoo(86996, 64, 2).to(device)  # Ensure to send the model to the correct device
skipgram_model.load_state_dict(checkpoint['model_state_dict'])
skipgram_model.eval()  # Set the model to evaluation mode if you're not training it again

# Tower RNN Models
tower_one = TowerOneRNN().to(device)
tower_two = TowerTwoRNN().to(device)

# Triplet margin loss with cosine distance
triplet_loss_fn = nn.TripletMarginWithDistanceLoss(
    distance_function=lambda x, y: 1.0 - nn.functional.cosine_similarity(x, y),
    margin=1.0,
    reduction='mean'
)

# Define optimizer for the tower models
optimizer = optim.Adam(list(tower_one.parameters()) + list(tower_two.parameters()), lr=0.001)

# Define query batch size
query_batch_size = 128  # Number of queries to process in a batch

# Define number of epochs
num_epochs = 30  # Example

# Track model and hyperparameters in W&B
wandb.watch([tower_one, tower_two], log="all")  # Log gradients and model weights

# Training loop with batched queries
for epoch in range(num_epochs):
    # Group by 'query' to handle variable number of positives and negatives per query
    query_groups = list(results.groupby('query'))

    # Iterate through the dataset in batches of 'query_batch_size' queries
    for q_batch_start in tqdm(range(0, len(query_groups), query_batch_size), desc=f"Epoch {epoch + 1}/{num_epochs}"):
        query_batch = query_groups[q_batch_start:q_batch_start + query_batch_size]

        all_anchor_embeddings = []
        all_positive_embeddings = []
        all_negative_embeddings = []

        # Process each query group in the current batch
        for query, group in query_batch:
            # Tokenize the queries, passage_text, and negative_sample using your updated_vocab or reverse_vocab
            query_tokens = tokenize_titles(group['query'].tolist(), reverse_vocab)
            positive_tokens = tokenize_titles(group['passage_text'].tolist(), reverse_vocab)
            negative_tokens = tokenize_titles(group['negative_sample'].tolist(), reverse_vocab)

            # Get embeddings for each group
            anchor_embeddings = get_embeddings_for_titles(query_tokens, skipgram_model)
            positive_embeddings = get_embeddings_for_titles(positive_tokens, skipgram_model)
            negative_embeddings = get_embeddings_for_titles(negative_tokens, skipgram_model)

            # Append the embeddings to the lists
            all_anchor_embeddings.append(torch.tensor(anchor_embeddings))
            all_positive_embeddings.append(torch.tensor(positive_embeddings))
            all_negative_embeddings.append(torch.tensor(negative_embeddings))

        # Pad sequences to the same length
            anchor_batch = pad_sequence(all_anchor_embeddings, batch_first=True).to(device)
            positive_batch = pad_sequence(all_positive_embeddings, batch_first=True).to(device)
            negative_batch = pad_sequence(all_negative_embeddings, batch_first=True).to(device)

            # Get lengths of the original sequences and move them to CPU
            anchor_lengths = torch.tensor([len(seq) for seq in all_anchor_embeddings]).cpu()  # Move to CPU
            positive_lengths = torch.tensor([len(seq) for seq in all_positive_embeddings]).cpu()  # Move to CPU
            negative_lengths = torch.tensor([len(seq) for seq in all_negative_embeddings]).cpu()  # Move to CPU

            # Forward pass through the two towers
            anchor_output = tower_one(anchor_batch, anchor_lengths)
            positive_output = tower_two(positive_batch, positive_lengths)
            negative_output = tower_two(negative_batch, negative_lengths)



        # Calculate triplet loss
        triplet_loss = triplet_loss_fn(anchor_output, positive_output, negative_output)

        # Backpropagation
        optimizer.zero_grad()
        triplet_loss.backward()
        optimizer.step()

        # Log the loss to W&B
        wandb.log({"epoch": epoch + 1, "triplet_loss": triplet_loss.item()})

    print(f"Epoch {epoch + 1}/{num_epochs} completed. Loss: {triplet_loss.item()}")

    # Optionally, save a model checkpoint after every epoch and log it to W&B
    torch.save({
        'model_state_dict': skipgram_model.state_dict(),
        'tower_one_state_dict': tower_one.state_dict(),
        'tower_two_state_dict': tower_two.state_dict(),
    }, f"model_1_checkpoint_epoch_{epoch+1}.pth")

    wandb.finish

print("Training complete.")
```
