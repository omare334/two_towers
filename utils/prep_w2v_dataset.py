import collections
import more_itertools
import torch
import numpy as np
from tqdm import tqdm
from text_utils import preprocess_text

with open("dataset/text8") as f:
    text8: str = f.read()


corpus: list[str] = preprocess_text(text8)
print("Created main corpus")


def create_lookup_tables(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    word_counts = collections.Counter(words)
    vocab = sorted(word_counts, key=lambda k: word_counts.get(k), reverse=True)
    int_to_vocab = {ii + 1: word for ii, word in enumerate(vocab)}
    int_to_vocab[0] = "<PAD>"
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
    return vocab_to_int, int_to_vocab


words_to_ids, ids_to_words = create_lookup_tables(corpus)
print("Created lookup tables")

tokens = [words_to_ids[word] for word in corpus]
token_counts = collections.Counter(tokens)
total_token_len = len(tokens)


# Subsample corpus
def subsample_corpus(corpus_tokens, threshold):
    token_counts = collections.Counter(corpus_tokens)

    def keep_probability(token) -> float:
        return 1 - np.sqrt((total_token_len * threshold) / token_counts[token])

    tqdm_tkn = tqdm(corpus_tokens, desc="Subsampling")

    subsampled = [
        token for token in tqdm_tkn if keep_probability(token) > np.random.ranf()
    ]
    return subsampled


# print("Saubsampling...")
tokens_sub = subsample_corpus(corpus_tokens=tokens, threshold=1e-5)
# print("Finished subsampling")

context_window = 5
if context_window % 2 == 0:
    raise Exception(
        f"Context Window must be an odd number, currently: {context_window}"
    )
center_i = context_window // 2

windows = more_itertools.windowed(tokens_sub, context_window)

wind_tq = tqdm(windows, desc="Sliding window", total=len(tokens_sub))
neg_sample_count = 20

inputs = []
targets = []
neg_samples = []
for tkn_wind in wind_tq:
    inputs.append(tkn_wind[center_i])
    targets.append(tkn_wind[:center_i] + tkn_wind[center_i + 1 :])
    negs = [
        words_to_ids[corpus[id]]
        for id in np.random.randint(0, total_token_len, neg_sample_count)
    ]
    neg_samples.append(negs)


input_tensor = torch.LongTensor(inputs)
target_tensor = torch.LongTensor(targets)
negs_tensor = torch.LongTensor(neg_samples)

save_path = "dataset/processed/text8_set.pth"

torch.save((input_tensor, target_tensor, negs_tensor), save_path)
