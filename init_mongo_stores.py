import openai
from llama_index import Document, VectorStoreIndex
import os
import logging

import myutils

global_conf = myutils.global_light.config
INDEX_ID = global_conf.get('default', 'INDEX_ID')

assert os.getenv("OPENAI_API_KEY") is not None, "please set openai key!"
assert os.getenv("all_proxy") is not None, "please set proxy!"
assert INDEX_ID, "no index id set!"

openai.api_key = os.getenv("OPENAI_API_KEY")
# ======= end of init ===============

storage_context = myutils.get_mongo_storage()
documents = [Document(text="the first document.", doc_id="##Zero")]

# create new storages
index = VectorStoreIndex.from_documents(documents=documents,
                                        storage_context=storage_context)
index.set_index_id(INDEX_ID)
storage_context.persist()

logging.info("llama indices initialized as '%s'.", INDEX_ID)
print(''' 
# to create kNN index on mongodb field, you need change the default index 
of the "vectors" collection to following:
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "dimensions": 1536,
        "similarity": "cosine",
        "type": "knnVector"
      }
    }
  }
}
''')
