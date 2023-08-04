import pandas as pd
import openai
import os
import json

from openai.embeddings_utils import get_embedding

os.environ['all_proxy'] = 'socks5h://localhost:1086'
EMBEDDING_MODEL = "text-embedding-ada-002"
COMPLETIONS_MODEL = "gpt-4"
# openai.api_key = os.getenv("OPENAI_API_KEY")


def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.

    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.content, EMBEDDING_MODEL) for idx, r in df.iterrows()
    }


df = pd.read_csv('legislation.csv')
#print(f"before head {len(df)} rows in the data.")
#df.head()
df = df.set_index(['title', 'heading'])
print(f"{len(df)} rows in the data.")
document_embeddings = compute_doc_embeddings(df)
# An example embedding:
# example_entry = list(document_embeddings.items())[0]
# print(f"{example_entry[0]} : {example_entry[1][:5]}... ({len(example_entry[1])} entries)")

with open('embeddings.json', 'w') as f:
    json.dump({str(k): v for k, v in document_embeddings.items()}, f)

print(f"{len(document_embeddings)} embeddings saved to local!")


