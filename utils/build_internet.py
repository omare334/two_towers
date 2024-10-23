import datasets
import pickle
from tqdm import tqdm
from pathlib import Path

internet_dir = Path(f"dataset/internet")
internet_dir.mkdir(parents=True, exist_ok=True)
INTERNET_FILEPATH = internet_dir / "all_docs"

if __name__ == "__main__":
    dataset = datasets.load_dataset("microsoft/ms_marco", "v1.1")

    splits = ["train", "test", "validation"]
    all_documents: set[str] = set()

    for split in splits:

        passages = dataset[split]["passages"]  # [:10000]
        for passage in tqdm(passages, desc=split + " passages"):
            all_documents.update(set(passage["passage_text"]))

    with open(INTERNET_FILEPATH, "w") as f:
        for index, doc in tqdm(
            enumerate(all_documents),
            desc="Writing into document",
            total=len(all_documents),
        ):
            f.write(f"{doc}\n")
