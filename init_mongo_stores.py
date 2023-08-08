import openai
import pymongo
from llama_index import StorageContext, Document, VectorStoreIndex
from llama_index.storage.docstore import MongoDocumentStore
from llama_index.storage.index_store import MongoIndexStore
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
import os
import logging

import config
import myutils

global_conf = config.initialize_config()
INDEX_ID = global_conf.get('default', 'INDEX_ID')

assert os.getenv("OPENAI_API_KEY") is not None, "please set openai key!"
assert os.getenv("all_proxy") is not None, "please set proxy!"
assert INDEX_ID, "no index id set!"

openai.api_key = os.getenv("OPENAI_API_KEY")
# ======= end of init ===============

# mongodb atlas vector store
# mongodb index store and document store
storage_context = myutils.get_mongo_storage()
documents = [Document(text="the first document.", doc_id="##Zero")]

# create new storages
index = VectorStoreIndex.from_documents(documents=documents,
                                        storage_context=storage_context)
index.set_index_id(INDEX_ID)
storage_context.persist()
logging.info("llama indices initialized as '%s'.", INDEX_ID)