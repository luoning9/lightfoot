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
MONGODB_URI = "mongodb+srv://llama:oBJhFhXfVqUVm5jT@cluster0.emjnc5c.mongodb.net/?retryWrites=true&w=majority"

# init Chroma collection
chroma_client = chromadb.PersistentClient(path="./storage/chromadb/")
chroma_collection = chroma_client.create_collection(name="docs_first",
                                                    get_or_create=True)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# mongodb index store
# create (or load) index store
index_store = MongoIndexStore.from_uri(uri=MONGODB_URI)
doc_store = MongoDocumentStore.from_uri(uri=MONGODB_URI)

storage_context = StorageContext.from_defaults(
    docstore=doc_store,
    index_store=index_store,
    vector_store=vector_store
)

index = load_index_from_storage(storage_context, index_id=INDEX_ID)
documents = SimpleDirectoryReader('examples/paul_graham_essay/data',
                                  filename_as_id=True).load_data()
for doc in documents:
    logging.debug("loaded doc %s", doc.text[:40]+"..." if len(doc.text)>43 else doc.text )
    doc_id = doc.get_doc_id()
    print(f"doc exists? {index.docstore.get_document_hash(doc_id)}")
updated = index.refresh_ref_docs(documents)
print(f"update status {updated}")
storage_context.persist()

# query_engine = index.as_query_engine()
# response = query_engine.query("What did the author do when he was young?")
# print(response)
