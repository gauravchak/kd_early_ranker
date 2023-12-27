import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline_early_ranker import MultitaskEarlyRanker


class KDAuxMultitaskEarlyRanker(MultitaskEarlyRanker):
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
        super(KDAuxMultitaskEarlyRanker, self).__init__(
            input_size_user,
            num_user_embeddings,
            num_items,
            hidden_size,
            embedding_dim,
            num_user_tasks
        )


# Example usage:
input_size_user = 10
num_items = 1000  # Example number of items
hidden_size = 20
embedding_dim = 64
num_user_tasks = 3  # 3 Tasks
num_user_embeddings = 3  # Learning multiple modes/user interests

# Instantiate KDAuxMultitaskEarlyRanker with tasks = 2*T
model = KDAuxMultitaskEarlyRanker(
    input_size_user=input_size_user,
    num_user_embeddings=num_user_embeddings,
    num_items=num_items,
    hidden_size=hidden_size,
    embedding_dim=embedding_dim,
    num_user_tasks=2*num_user_tasks
)

# Example input tensors
user_features = torch.randn((32, input_size_user))  # Batch size of 32
item_ids = torch.randint(0, num_items, (32,))  # item IDs
hard_labels = torch.randint(0, 2, (32, num_user_tasks))  # [B, T]
soft_labels = torch.rand((32, num_user_tasks))  # [B, T]
labels = torch.concat([hard_labels, soft_labels], dim=-1)  # [B, 2*T]

# Forward pass for training
loss = model.train_forward(user_features, item_ids, labels)

print("Training Loss:", loss.item())
