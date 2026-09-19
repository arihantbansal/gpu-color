from torch import nn


def ngram(tokens: str, n=3) -> list[str]:
    return [tokens[i : i + n] for i in range(len(tokens) - n + 1)]


def tokenize(phrase: str, buckets: int = 4096) -> list[int]:
    with_boundary_markers = [
        ng for y in phrase.lower().split() for ng in ngram(f"<{y}>")
    ]

    hash_seed = 2166136261
    tokens = []
    for ng in with_boundary_markers:
        fragment = ng.encode("utf-8")
        id = hash_seed
        for char in fragment:
            id ^= char
            id *= 16777619
            id = id & 0xFFFFFFFF

        tokens.append(id % buckets)

    return tokens


class ColorModel(nn.Module):
    def __init__(self, buckets: int = 4096, dim: int = 32, hidden: int = 64):
        super().__init__()

        self.embedding_bag = nn.EmbeddingBag(buckets, dim, mode="mean")

        self.fc1 = nn.Linear(dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 3)

    def forward(self, input, offsets=None):
        average_embedding = self.embedding_bag(input, offsets)

        x = self.fc1(average_embedding)
        x = self.relu(x)
        x = self.fc2(x)

        return x
