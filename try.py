from llama_index import SimpleDirectoryReader, StorageContext, load_index_from_storage, GPTVectorStoreIndex
from llama_index.storage.docstore import MongoDocumentStore, SimpleDocumentStore
from llama_index.storage.index_store import MongoIndexStore, SimpleIndexStore
from llama_index.vector_stores import ChromaVectorStore, SimpleVectorStore
import os
import chromadb
import logging
import sys
from chromadb.config import Settings
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

os.environ['all_proxy'] = 'socks5h://127.0.0.1:1086'
assert os.getenv("OPENAI_API_KEY"), "please set openai key."

PERSIST_DIR = "./storage"
INDEX_ID = "paul"

# init Chroma collection
chroma_client = chromadb.PersistentClient(path="./storage/chromadb/")
chroma_collection = chroma_client.create_collection(name="docs_first",
                                                    get_or_create=True)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(
    docstore=SimpleDocumentStore.from_persist_dir(persist_dir=PERSIST_DIR),
    index_store=SimpleIndexStore.from_persist_dir(persist_dir=PERSIST_DIR),
#    vector_store = SimpleVectorStore.from_persist_dir(persist_dir=PERSIST_DIR)
    vector_store = vector_store
)
index = load_index_from_storage(storage_context, index_id=INDEX_ID)
documents = SimpleDirectoryReader('examples/paul_graham_essay/data',
                                  filename_as_id=True).load_data()
for doc in documents:
    logging.debug("loaded doc %s", doc.text[:40]+"..." if len(doc.text)>43 else doc.text )
updated = index.refresh_ref_docs(documents)
print(f"update status {updated}")

# create new storages
# index = GPTVectorStoreIndex.from_documents(documents=documents,
#                                           storage_context=storage_context)
# index.set_index_id(INDEX_ID)
storage_context.persist()


# query_engine = index.as_query_engine()
# response = query_engine.query("What did the author do when he was young?")
# print(response)
