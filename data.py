from datasets import load_dataset
import pandas as pd
import random
import ast

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
        result.append(random_doc)
    return result

# Returns a tuple of (query, positive_docs, negative_docs)
def get_training_data():
    df = get_ms_marco_data()
    random_row = df.sample(n=1)
    query = random_row["query"].values[0]
    pos_docs = random_row["passage_text"].values[0]
    neg_docs = get_random_docs(len(pos_docs), df)
    return (query, pos_docs, neg_docs)

triple = get_training_data()
print(triple)
