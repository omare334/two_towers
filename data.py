from datasets import load_dataset
import pandas as pd
import random
import ast
import pickle

from utils.tokeniser import Tokeniser

print("Script started")

# You should run this once wuth fromHuggingFace true
#  so that it gets downloaded to the data folder
def load_data_as_df(fromHuggingFace=False):
    if fromHuggingFace:
        print("Loading dataset from HuggingFace")
        ds = load_dataset("microsoft/ms_marco", "v1.1")
        df_train = pd.DataFrame(ds["train"])
        df_train.to_csv('ms_marco_train.csv')
    else:
        df_train = pd.read_csv("ms_marco_train.csv")
    return df_train

# Returns a dataframe with columns "passage_text" and "query"
def get_ms_marco_data():
    df = load_data_as_df()
    print("Processing passages and queries")
    df['passages'] = df['passages'].apply(ast.literal_eval)
    df["passage_text"] = df["passages"].apply(lambda x: x["passage_text"])
    data = df[["passage_text", "query"]]
    return data

# Returns a list of n random docs
def get_random_docs(n, df, exclude_docs):
    result = []
    for i in range(n):
        random_row = df.sample(n=1)
        docs = random_row["passage_text"].values[0]
        random_doc = random.sample(docs, 1)
        if random_doc not in exclude_docs:
            result.append(random_doc[0])
    return result

# Returns a tuple of (query, positive_docs, negative_docs)
def get_training_data(df):
    # df = get_ms_marco_data()
    random_row = df.sample(n=1)
    query = random_row["query"].values[0]
    pos_docs = random_row["passage_text"].values[0]
    if len(pos_docs) == 0:
        return None
    neg_docs = get_random_docs(len(pos_docs), df, pos_docs)
    return (query, pos_docs, neg_docs)

# Returns a tuple of (query, positive_docs, negative_docs)
def get_training_data_tokenised(df):
    data = get_training_data(df)
    if data is None:
        return None
    query, pos, neg = data
    tokeniser = Tokeniser()
    query_tokens = tokeniser.tokenise(query)
    pos_tokens = [tokeniser.tokenise(doc) for doc in pos]
    neg_tokens = [tokeniser.tokenise(doc) for doc in neg]
    return (query_tokens, pos_tokens, neg_tokens)

def process_and_pickle_data(output_file="dataset/training_data.pkl", num_samples=1000):
    df = get_ms_marco_data()
    
    # Store all (query, positive_docs, negative_docs) tuples
    data_tuples = []
    for _ in range(num_samples):
        tokenized_data = get_training_data_tokenised(df)
        if tokenized_data:
            data_tuples.append(tokenized_data)
    
    # Save the data tuples to a pickle file
    with open(output_file, "wb") as f:
        pickle.dump(data_tuples, f)
    print(f"Data saved to {output_file}")

