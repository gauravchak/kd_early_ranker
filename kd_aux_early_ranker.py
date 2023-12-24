import torch
import torch.nn as nn
import torch.nn.functional as F

class KDAuxMultitaskEarlyRanker(nn.Module):
    def __init__(
        self,
        input_size_user: int,
        num_items: int,
        hidden_size: int,
        embedding_dim: int,
        num_user_tasks: int
    ) -> None:
        """
        params:
        input_size_user: user_features dimension
        num_items: item embedding lookup table size.
        hidden_size: hidden layer size in user features MLP
        embedding_dim: dimension of user and item embeddings
        num_user_tasks: the number of user labels to compute predictions for
        """
        super(KDAuxMultitaskEarlyRanker, self).__init__()

        # Define MLP layers for user features
        self.user_mlp = nn.Sequential(
            nn.Linear(input_size_user, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_dim)
        )

        # Define an embedding layer for item IDs
        self.item_embedding_layer = nn.Embedding(num_items, embedding_dim)

        self.mtml = nn.Sequential(
            nn.Linear(
                2 * embedding_dim, 
                2 * num_user_tasks  # for both hard and soft labels
            )
        )

    def forward(
        self,
        user_features,  # [B, input_size_user]
        item_ids,  # [B]
    ) -> torch.Tensor:
        # Pass user features through MLP
        user_embeddings = self.user_mlp(user_features)

        # Get item embeddings using the embedding layer
        item_embeddings = self.item_embedding_layer(item_ids)

        # Concatenate user and item embeddings
        concatenated_embeddings = torch.cat(
            [user_embeddings, item_embeddings],
            dim=1
        )

        # Pass through MTML to compute task logits
        task_logits = self.mtml(concatenated_embeddings)  # [B, 2*T]

        return task_logits

    def train_forward(
        self,
        user_features,  # [B, input_size_user]
        item_ids,  # [B]
        labels  # [B, 2 * num_user_tasks]
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
num_user_tasks = 3  # 3 Tasks

# Instantiate the multitask early ranker
model = KDAuxMultitaskEarlyRanker(
    input_size_user, 
    num_items, 
    hidden_size, 
    embedding_dim, 
    num_user_tasks
)

# Example input tensors
user_features = torch.randn((32, input_size_user))  # Batch size of 32
item_ids = torch.randint(0, num_items, (32,))  # item IDs
hard_labels = torch.randint(0, 2, (32, num_user_tasks))
soft_labels = torch.rand((32, num_user_tasks))
labels = torch.concat([hard_labels, soft_labels], dim=-1)

# Forward pass for training
loss = model.train_forward(user_features, item_ids, labels)

print("Training Loss:", loss.item())
