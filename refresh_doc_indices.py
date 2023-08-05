import pymongo
from llama_index import SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.storage.docstore import MongoDocumentStore
from llama_index.storage.index_store import MongoIndexStore
import os
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

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

index = load_index_from_storage(storage_context, index_id=INDEX_ID)
documents = SimpleDirectoryReader('examples/paul_graham_essay/data',
                                  filename_as_id=True).load_data()
for doc in documents:
    #    logging.debug("loaded doc %s", doc.text[:40]+"..." if len(doc.text)>43 else doc.text )
    doc_id = doc.get_doc_id()
    print(f"doc {doc_id} exists? {index.docstore.get_document_hash(doc_id)}")
updated = index.refresh_ref_docs(documents)
print(f"update status {updated}")
storage_context.persist()
