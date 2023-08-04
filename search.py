import os
import json
import tiktoken
import ast
import openai
import numpy as np
import pandas as pd
from openai.embeddings_utils import get_embedding


os.environ['all_proxy'] = 'socks5h://localhost:1086'
EMBEDDING_MODEL = "text-embedding-ada-002"
COMPLETIONS_MODEL = "gpt-4"
openai.api_key = os.getenv("OPENAI_API_KEY")


## This code was written by OpenAI: https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb


def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.

    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return float(np.dot(np.array(x), np.array(y)))


def order_by_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query, EMBEDDING_MODEL)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> tuple[list[str], int]:
    """
    Fetch relevant
    """
    most_relevant_document_sections = order_by_similarity(question, context_embeddings)

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.
        document_section = df.loc[section_index]

        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))

    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))

    return chosen_sections, chosen_sections_len


def answer_with_gpt_4(
        query: str,
        df: pd.DataFrame,
        document_embeddings: dict[(str, str), np.array],
        show_prompt: bool = False
) -> str:
    messages = [
        {"role": "system",
         "content": "You are a GDPR chatbot, only answer the question by using the provided context. If your are unable to answer the question using the provided context, say 'I don't know'"}
    ]
    prompt, section_length = construct_prompt(
        query,
        document_embeddings,
        df
    )
    if show_prompt:
        print(prompt)

    context = ""
    for article in prompt:
        context = context + article

    context = context + '\n\n --- \n\n + ' + query

    messages.append({"role": "user", "content": context})
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )

    return '\n' + response['choices'][0]['message']['content'], section_length


MAX_SECTION_LEN = 2000
SEPARATOR = "\n* "
ENCODING = "cl100k_base"  # encoding for text-davinci-003

articles = pd.read_csv('legislation.csv')
articles = articles.set_index(['title', 'heading'])

# load embeddings database to a dict
with open('embeddings.json', 'r') as f:
    doc_embeddings = json.load(f)
# Convert keys back to tuples
doc_embeddings = {ast.literal_eval(k): v for k, v in doc_embeddings.items()}

# exit(1)

result = order_by_similarity("Can the commission implement acts for exchanging information?", doc_embeddings)[:5]
print(result)

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))
print(f"Context separator contains {separator_len} tokens")

print(answer_with_gpt_4("How long is my information stored ?", articles, doc_embeddings))

