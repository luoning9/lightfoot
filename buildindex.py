import pandas as pd
import openai
import os

os.environ['all_proxy'] = 'socks5://localhost:1086'

df = pd.read_csv('legislation.csv')
df.head()
df = df.set_index(["title", "heading"])
print(f"{len(df)} rows in the data.")
# print(df.iloc[33].to_markdown())

EMBEDDING_MODEL = "text-embedding-ada-002"
COMPLETIONS_MODEL = "gpt-4"


## This code was written by OpenAI: https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb

def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    print(text)
    result = openai.Embedding.create(
        model=model,
        input=text
    )
    print(result["data"])
    return result["data"][0]["embedding"]


def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.

    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.content) for idx, r in df.iterrows()
    }


document_embeddings = compute_doc_embeddings(df)

# An example embedding:
example_entry = list(document_embeddings.items())[0]
print(f"{example_entry[0]} : {example_entry[1][:5]}... ({len(example_entry[1])} entries)")