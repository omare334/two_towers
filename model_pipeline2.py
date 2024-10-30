import pandas as pd
import torch
import numpy as np
import gensim.downloader as api
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import wandb
from sklearn.model_selection import train_test_split
import os

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Split the dataset into training, validation, and test sets (80-10-10)
print("Loading cleaned MS MARCO data...")
results_df = pd.read_csv('results_negative.csv')  # Replace with your dataset path

# Using the full dataset for training
print("Splitting dataset into training (80%), validation (10%), and test (10%)...")
train_df, temp_df = train_test_split(results_df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Training set: {len(train_df)} samples")
print(f"Validation set: {len(val_df)} samples")
print(f"Test set: {len(test_df)} samples")

# Initialize Wandb for experiment tracking
wandb.init(project="two-tower-rnn", entity="fxarya")

# Load the Word2Vec model
print("Loading Word2Vec model...")
word_vectors = api.load("word2vec-google-news-300")
print("Word2Vec model loaded successfully.")

# Create an Embedding Layer for Unseen Words
unknown_embedding_dim = 300
num_unknown_tokens = 1000  # Arbitrary number for unseen words
unseen_word_embedding_layer = nn.Embedding(num_unknown_tokens, unknown_embedding_dim)

# Create a dictionary to store indices for unseen words
unseen_word_dict = {}
next_unknown_index = 0
# Use global keyword in the function to properly modify next_unknown_index

# Function to convert a text to a vector using Word2Vec
def txt2vec(txt):
    global next_unknown_index
    # Ensure txt is a string
    if not isinstance(txt, str):
        txt = str(txt)
    # Retrieve embeddings for each word in the text
    word_embeddings = []
    for word in txt.lower().split():
        if word in word_vectors:
            word_embeddings.append(torch.tensor(word_vectors[word]).to(device))
        else:
            # Use a consistent embedding for each unseen word
            if word not in unseen_word_dict:
                if next_unknown_index < num_unknown_tokens:
                    unseen_word_dict[word] = next_unknown_index
                    next_unknown_index += 1
                else:
                    unseen_word_dict[word] = 0  # Reuse index if exceeding max unknown tokens
            # Append the embedding for the unseen word by indexing the embedding layer
            word_index = unseen_word_dict[word]
            word_embeddings.append(unseen_word_embedding_layer(torch.tensor(word_index, device=device)).squeeze(0).to(device))
    
    if len(word_embeddings) == 0:
        # If no words were found, use the average embedding of all known word vectors
        word_embeddings = [word_vectors.vectors.mean(axis=0)]
    
    # Convert list of embeddings to a single numpy array, then to a tensor
    word_embeddings_np = np.array([embedding.detach().cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding for embedding in word_embeddings], dtype=np.float32)
    return torch.tensor(word_embeddings_np, dtype=torch.float).mean(dim=0)

# Define the Two-Tower Model including embedding for unseen words
class TwoTowerModel(nn.Module):
    def __init__(self, embedding_dim=300, hidden_dim=128):
        super(TwoTowerModel, self).__init__()
        # Define the query tower
        self.query_tower = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Define the document tower (similar structure but independent)
        self.doc_tower = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, query_vec, doc_vec):
        # Process through each tower
        query_embed = self.query_tower(query_vec)
        doc_embed = self.doc_tower(doc_vec)
        return query_embed, doc_embed

# Initialize the model, including the unseen word embedding layer, and move it to the specified device
print("Initializing the model...")

# Move the unseen word embedding layer to the specified device
unseen_word_embedding_layer = unseen_word_embedding_layer.to(device)
model = TwoTowerModel().to(device)
print("Model initialized successfully.")

# Apply embeddings to 'query', 'positive', and 'negative' columns in the DataFrames
print("Applying embeddings to 'query', 'positive', and 'negative' columns for training, validation, and test data...")
for df_name, df in zip(['train', 'validation', 'test'], [train_df, val_df, test_df]):
    print(f"Processing {df_name} data...")
    df = df.copy()  # Make an explicit copy to avoid SettingWithCopyWarning
    df.loc[:, 'query_vec'] = df['query'].apply(txt2vec)
    df.loc[:, 'positive_vec'] = df['passage_text'].apply(txt2vec)
    df.loc[:, 'negative_vec'] = df['negative_sample'].apply(txt2vec)
    
    # Save the embeddings to disk for later use
    embedding_file = f"{df_name}_embeddings.pt"
    torch.save({
        'query_vecs': torch.stack(df['query_vec'].tolist()),
        'positive_vecs': torch.stack(df['positive_vec'].tolist()),
        'negative_vecs': torch.stack(df['negative_vec'].tolist())
    }, embedding_file)
    print(f"{df_name.capitalize()} embeddings saved to {embedding_file}.")
    
    # Ensure that the modified DataFrame is updated
    if df_name == 'train':
        train_df = df
    elif df_name == 'validation':
        val_df = df
    elif df_name == 'test':
        test_df = df
    
    print(f"{df_name.capitalize()} embeddings applied successfully.")

print("All embeddings applied and saved successfully.")

# Create DataLoader for training, validation, and test sets
def create_data_loader(df, batch_size=64):  # Use a larger batch size for full dataset training
    query_vectors = torch.stack(df['query_vec'].tolist())
    positive_vectors = torch.stack(df['positive_vec'].tolist())
    negative_vectors = torch.stack(df['negative_vec'].tolist())

    dataset = TensorDataset(query_vectors, positive_vectors, negative_vectors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_loader = create_data_loader(train_df)
val_loader = create_data_loader(val_df)
test_loader = create_data_loader(test_df, batch_size=64)

# Define the Triplet Loss
triplet_loss = nn.TripletMarginLoss(margin=1.0)

# Initialize the optimizer for the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Training loop
print("Starting training loop...")
num_epochs = 5  # Increase the number of epochs for full training

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (query_batch, positive_batch, negative_batch) in enumerate(train_loader):
        # Add gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        query_batch, positive_batch, negative_batch = query_batch.to(device), positive_batch.to(device), negative_batch.to(device)

        # Ensure unseen word embedding layer is also in training mode
        unseen_word_embedding_layer.train()
        
        # Forward pass
        query_embed, positive_embed = model(query_batch, positive_batch)
        _, negative_embed = model(query_batch, negative_batch)

        # Compute loss
        loss = triplet_loss(query_embed, positive_embed, negative_embed)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

        # Log the loss for each batch to Wandb
        wandb.log({"train_batch_loss": loss.item(), "epoch": epoch + 1, "batch_idx": batch_idx})

    # Average epoch loss
    avg_epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")
    wandb.log({"train_epoch_loss": avg_epoch_loss, "epoch": epoch + 1})

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (query_batch, positive_batch, negative_batch) in enumerate(val_loader):
            query_batch, positive_batch, negative_batch = query_batch.to(device), positive_batch.to(device), negative_batch.to(device)

            query_embed, positive_embed = model(query_batch, positive_batch)
            _, negative_embed = model(query_batch, negative_batch)

            loss = triplet_loss(query_embed, positive_embed, negative_embed)
            val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        wandb.log({"val_loss": avg_val_loss, "epoch": epoch + 1})

print("Training complete.")

# Testing on the entire test dataset
print("Testing on the entire test dataset...")
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    total_pos_similarity = 0.0
    total_neg_similarity = 0.0
    num_batches = 0

    for batch_idx, (query_batch, positive_batch, negative_batch) in enumerate(test_loader):
        # Retrieve original queries, positives, and negatives for logging
        original_queries = test_df.iloc[batch_idx * 64 : batch_idx * 64 + len(query_batch)]['query'].tolist()
        original_positives = test_df.iloc[batch_idx * 64 : batch_idx * 64 + len(positive_batch)]['passage_text'].tolist()
        original_negatives = test_df.iloc[batch_idx * 64 : batch_idx * 64 + len(negative_batch)]['negative_sample'].tolist()
        
        # Print original text examples for the current batch
        for i in range(len(original_queries)):
            print(f"Test Batch - Query: {original_queries[i]}")
            print(f"Positive: {original_positives[i]}")
            print(f"Negative: {original_negatives[i]}")
        # Move the batch to the appropriate device
        query_batch, positive_batch, negative_batch = query_batch.to(device), positive_batch.to(device), negative_batch.to(device)
        
        # Get embeddings from the model
        query_embed, positive_embed = model(query_batch, positive_batch)
        _, negative_embed = model(query_batch, negative_batch)
        
        # Calculate cosine similarities
        pos_similarity = torch.cosine_similarity(query_embed, positive_embed, dim=1).mean()
        neg_similarity = torch.cosine_similarity(query_embed, negative_embed, dim=1).mean()
        
        # Accumulate similarities for final summary
        total_pos_similarity += pos_similarity.item()
        total_neg_similarity += neg_similarity.item()
        num_batches += 1
        
        # Print similarity results for current batch
        print(f"Batch {batch_idx + 1} - Positive similarity: {pos_similarity.item():.4f}, Negative similarity: {neg_similarity.item():.4f}")
        # Log test metrics to Wandb
        wandb.log({"test_positive_similarity": pos_similarity.item(), "test_negative_similarity": neg_similarity.item(), "batch_idx": batch_idx + 1})

    # Calculate and print average similarities over all batches
    avg_pos_similarity = total_pos_similarity / num_batches
    avg_neg_similarity = total_neg_similarity / num_batches
    print(f"Average Test - Positive similarity: {avg_pos_similarity:.4f}, Negative similarity: {avg_neg_similarity:.4f}")
    # Log average test metrics to Wandb
    wandb.log({"avg_test_positive_similarity": avg_pos_similarity, "avg_test_negative_similarity": avg_neg_similarity})

# Testing on a single batch of 2 examples from the test batch data
print("Testing on a single batch of 2 examples from the test batch data...")
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for batch_idx, (query_batch, positive_batch, negative_batch) in enumerate(test_loader):
        # Retrieve original queries, positives, and negatives for logging
        original_queries = test_df.iloc[batch_idx * 2 : batch_idx * 2 + len(query_batch)]['query'].tolist()
        original_positives = test_df.iloc[batch_idx * 2 : batch_idx * 2 + len(positive_batch)]['passage_text'].tolist()
        original_negatives = test_df.iloc[batch_idx * 2 : batch_idx * 2 + len(negative_batch)]['negative_sample'].tolist()
        
        # Print original text examples for the current batch
        for i in range(len(original_queries)):
            print(f"Test Batch - Query: {original_queries[i]}")
            print(f"Positive: {original_positives[i]}")
            print(f"Negative: {original_negatives[i]}")
        # Move the batch to the appropriate device
        query_batch, positive_batch, negative_batch = query_batch.to(device), positive_batch.to(device), negative_batch.to(device)
        
        # Get embeddings from the model
        query_embed, positive_embed = model(query_batch, positive_batch)
        _, negative_embed = model(query_batch, negative_batch)
        
        # Calculate cosine similarities
        pos_similarity = torch.cosine_similarity(query_embed, positive_embed, dim=1).mean()
        neg_similarity = torch.cosine_similarity(query_embed, negative_embed, dim=1).mean()
        
        # Print similarity results
        print(f"Test - Positive similarity: {pos_similarity.item():.4f}, Negative similarity: {neg_similarity.item():.4f}")
        # Log test metrics to Wandb
        wandb.log({"test_positive_similarity": pos_similarity.item(), "test_negative_similarity": neg_similarity.item()})
        # Process all test data without breaking
