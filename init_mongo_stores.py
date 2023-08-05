import pymongo
from llama_index import StorageContext, Document, VectorStoreIndex
from llama_index.storage.docstore import MongoDocumentStore
from llama_index.storage.index_store import MongoIndexStore
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
import os
import logging

import config

global_conf = config.initialize_config()
MONGODB_URI = global_conf.get('default', 'MONGODB_URI')
INDEX_ID = global_conf.get('default', 'INDEX_ID')

assert os.getenv("OPENAI_API_KEY"), "please set openai key!"
assert os.getenv("all_proxy"), "please set proxy!"
assert MONGODB_URI, "no mongodb uri set!"
assert INDEX_ID, "no index id set!"
# ======= end of init ===============

# mongodb atlas vector store
mongo_client = pymongo.MongoClient(MONGODB_URI)
vector_store = MongoDBAtlasVectorSearch(mongo_client)
# mongodb index store and document store
index_store = MongoIndexStore.from_uri(uri=MONGODB_URI)
doc_store = MongoDocumentStore.from_uri(uri=MONGODB_URI)

storage_context = StorageContext.from_defaults(
    docstore=doc_store,
    index_store=index_store,
    vector_store=vector_store
)
documents = [Document(text="the first document.", doc_id="##Zero")]

# create new storages
index = VectorStoreIndex.from_documents(documents=documents,
                                        storage_context=storage_context)
index.set_index_id(INDEX_ID)
storage_context.persist()
logging.info("llama indices initialized at " + str(mongo_client.address))