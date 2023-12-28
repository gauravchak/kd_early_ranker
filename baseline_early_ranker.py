import torch
import torch.nn as nn
import torch.nn.functional as F


class MultitaskEarlyRanker(nn.Module):
    def __init__(
        self,
        input_size_user: int,
        num_user_embeddings: int,
        num_items: int,
        hidden_size: int,
        embedding_dim: int,
        num_user_tasks: int
    ) -> None:
        """
        params:
        input_size_user: (IU) user_features dimension
        num_user_embeddings: (E) multiple modes/user interests
        num_items: item embedding lookup table size.
        hidden_size: hidden layer size in user features MLP
        embedding_dim: (D) dimension of user and item embeddings
        num_user_tasks: (T) the number of user labels to predict
        """
        super(MultitaskEarlyRanker, self).__init__()
        self.num_user_embeddings = num_user_embeddings
        self.embedding_dim = embedding_dim
        self.num_user_tasks = num_user_tasks

        # Define MLP layers for user features
        self.user_mlp = nn.Sequential(
            nn.Linear(input_size_user, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_user_embeddings * embedding_dim)
        )

        # Define an embedding layer for item IDs
        self.item_embedding_layer = nn.Embedding(num_items, embedding_dim)

        self.mtml = nn.Sequential(
            nn.Linear(2 * embedding_dim + num_user_embeddings, num_user_tasks)
        )

    def forward(
        self,
        user_features,  # [B, IU]
        item_ids,  # [B]
    ) -> torch.Tensor:
        # Pass user features through MLP
        user_embeddings = self.user_mlp(user_features)  # [B, E * D]
        user_embeddings = user_embeddings.view(
            -1, self.num_user_embeddings, self.embedding_dim
        )  # [B, E, D]

        # Get item embeddings using the embedding layer
        item_embeddings = self.item_embedding_layer(item_ids)  # [B, D]

        # Compute dot product between user and item embeddings
        user_item_dots = torch.bmm(
            user_embeddings, item_embeddings.unsqueeze(-1)
        ).squeeze(-1)  # [B, E]
        normalized_user_item_dots = torch.nn.functional.softmax(user_item_dots, dim=1)  # [B, E]
        combined_user_emebdding = torch.bmm(
            user_embeddings.permute(0, 2, 1), 
            normalized_user_item_dots.unsqueeze(-1)
        ).squeeze(-1)  # [B, D]


        # Concatenate user and item embeddings
        concatenated_embeddings = torch.cat(
            [combined_user_emebdding, item_embeddings, user_item_dots],
            dim=1
        )  # [B, D+D+E]

        # Pass through MTML to compute task logits
        task_logits = self.mtml(concatenated_embeddings)  # [B, T]

        return task_logits

    def train_forward(
        self,
        user_features,  # [B, IU]
        item_ids,  # [B]
        labels  # [B, T]
    ) -> float:
        # Get task logits using forward method
        task_logits = self.forward(user_features, item_ids)

        # Compute binary cross-entropy loss
        cross_entropy_loss = F.binary_cross_entropy_with_logits(
            input=task_logits, target=labels.float(), reduction='sum'
        )

        return cross_entropy_loss

# Example usage:
input_size_user = 10
num_items = 1000  # Example number of items
hidden_size = 20
embedding_dim = 64
num_user_tasks = 4
num_user_embeddings = 3  # Learning multiple modes/user interests

# Instantiate the multitask early ranker
model = MultitaskEarlyRanker(
    input_size_user=input_size_user,
    num_user_embeddings=num_user_embeddings,
    num_items=num_items,
    hidden_size=hidden_size,
    embedding_dim=embedding_dim,
    num_user_tasks=num_user_tasks
)

# Example input tensors
user_features = torch.randn((32, input_size_user))  # Batch size of 32
item_ids = torch.randint(0, num_items, (32,))  # item IDs
labels = torch.randint(0, 2, (32, num_user_tasks))  # binary labels

# Forward pass for training
loss = model.train_forward(user_features, item_ids, labels)

print("Training Loss:", loss.item())
