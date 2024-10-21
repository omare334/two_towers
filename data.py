from datasets import load_dataset
import pandas as pd
import random
import ast

from utils.tokeniser import Tokeniser

# You should run this once wuth fromHuggingFace true
#  so that it gets downloaded to the data folder
def load_data_as_df(fromHuggingFace=False):
    if fromHuggingFace:
        ds = load_dataset("microsoft/ms_marco", "v1.1")
        df_train = pd.DataFrame(ds["train"])
        df_train.to_csv('data/ms_marco_train.csv')
    else:
        df_train = pd.read_csv("data/ms_marco_train.csv")
    return df_train

# Returns a dataframe with columns "passage_text" and "query"
def get_ms_marco_data():
    df = load_data_as_df()
    df['passages'] = df['passages'].apply(ast.literal_eval)
    df["passage_text"] = df["passages"].apply(lambda x: x["passage_text"])
    data = df[["passage_text", "query"]]
    return data

# Returns a list of n random docs
def get_random_docs(n, df):
    result = []
    for i in range(n):
        random_row = df.sample(n=1)
        docs = random_row["passage_text"].values[0]
        random_doc = random.sample(docs, 1)
        result.append(random_doc[0])
    return result

# Returns a tuple of (query, positive_docs, negative_docs)
def get_training_data():
    df = get_ms_marco_data()
    random_row = df.sample(n=1)
    query = random_row["query"].values[0]
    pos_docs = random_row["passage_text"].values[0]
    neg_docs = get_random_docs(len(pos_docs), df)
    return (query, pos_docs, neg_docs)

# Returns a tuple of (query, positive_docs, negative_docs)
def get_training_data_tokenised():
    (query, pos, neg) = get_training_data()
    tokeniser = Tokeniser()
    query_tokens = tokeniser.tokenise(query)
    pos_tokens = [tokeniser.tokenise(doc) for doc in pos]
    neg_tokens = [tokeniser.tokenise(doc) for doc in neg]
    return (query_tokens, pos_tokens, neg_tokens)

# triple = get_training_data_tokenised()
# print(triple)
