from llama_index import SimpleDirectoryReader, StorageContext, load_index_from_storage, GPTVectorStoreIndex
from llama_index.vector_stores import ChromaVectorStore, SimpleVectorStore
import os
import chromadb
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

os.environ['all_proxy'] = 'socks5h://127.0.0.1:1086'

assert os.getenv("OPENAI_API_KEY"), "please set openai key."
assert os.getenv("OPENAI_API_KEY"), "please set proxy"

PERSIST_DIR = "./storage"
INDEX_ID = "paul"

# init Chroma collection
chroma_client = chromadb.PersistentClient(path="./storage/chromadb/")
chroma_collection = chroma_client.create_collection(name="docs_first",
                                                    get_or_create=True)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(
    vector_store = vector_store
)
documents = SimpleDirectoryReader('examples/paul_graham_essay/data',
                                  filename_as_id=True).load_data()
# create new storages
index = GPTVectorStoreIndex.from_documents(documents=documents,
                                           storage_context=storage_context)
index.set_index_id(INDEX_ID)
storage_context.persist()
