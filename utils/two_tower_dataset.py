from torch.utils.data import Dataset

type Sequence = list[int]


class TwoTowerDataset(Dataset):
    def __init__(
        self, queries: list[Sequence], pos: list[Sequence], negs: list[Sequence]
    ):
        self.queries = queries
        self.pos = pos
        self.negs = negs

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx: int) -> tuple[Sequence, Sequence, Sequence]:
        return self.queries[idx], self.pos[idx], self.negs[idx]
