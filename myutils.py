import logging
import os
import sys
from configparser import ConfigParser

import openai
import pymongo
from llama_index import StorageContext
from llama_index.storage.docstore import MongoDocumentStore
from llama_index.storage.index_store import MongoIndexStore
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch


class GlobalObject:
    config: ConfigParser

    def __init__(self):
        # 初始化全局对象的属性
        parser = ConfigParser()
        parser.read("./private_config.ini")
        # set environment vars
        if os.getenv('all_proxy') is None:
            os.environ['all_proxy'] = parser.get('default', 'SOCKS_PROXY')
        if os.getenv('OPENAI_API_KEY') is None:
            os.environ['OPENAI_API_KEY'] = parser.get('default', 'OPENAI_API_KEY')

        assert os.getenv("OPENAI_API_KEY") is not None, "please set openai key!"
        assert os.getenv("all_proxy") is not None, "please set proxy!"
        openai.api_key = os.getenv("OPENAI_API_KEY")

        # set global vars
        self.config = parser

        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


# 全局对象的初始化
global_light = GlobalObject()

'''
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
'''


def clear_mongo_stores() -> None:
    config_parser = global_light.config
    uri = config_parser.get('mongodb', 'URI')
    assert uri is not None, 'no db uri specified!'

    client = pymongo.MongoClient(uri)
    db = client.get_database(config_parser.get('mongodb', 'DB_NAME'))
    if db is not None:
        names = db.list_collection_names()
        for cname in names:
            db.drop_collection(cname)
        logging.info("all collections in database %s dropped.", db.name)
    client.close()


def get_mongo_storage() -> StorageContext:
    config_parser = global_light.config
    uri = config_parser.get('mongodb', 'URI')
    db_name = config_parser.get('mongodb', 'DB_NAME')
    collection_name = config_parser.get('mongodb', 'VECTOR_COLLECTION')
    assert uri is not None, 'no db uri specified!'
    assert db_name is not None, 'no db name specified!'
    assert collection_name is not None, 'no vector collection name specified!'

    client = pymongo.MongoClient(uri)
    vector_store = MongoDBAtlasVectorSearch(client,
                                            db_name=db_name,
                                            collection_name=collection_name)
    # mongodb index store and document store
    index_store = MongoIndexStore.from_uri(uri=uri, db_name=db_name)
    doc_store = MongoDocumentStore.from_uri(uri=uri, db_name=db_name)
    storage_context = StorageContext.from_defaults(
        docstore=doc_store,
        index_store=index_store,
        vector_store=vector_store
    )
    return storage_context




def initialize_config() -> ConfigParser:
    global CONFIG
    if CONFIG is True:
        config = ConfigParser()
        config.read("./private_config.ini")
        # set environment vars
        if os.getenv('all_proxy') is None:
            os.environ['all_proxy'] = config.get('default','SOCKS_PROXY')
        if os.getenv('OPENAI_API_KEY') is None:
            os.environ['OPENAI_API_KEY'] = config.get('default', 'OPENAI_API_KEY')

        # set global vars
        CONFIG = config

        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    return CONFIG


CONFIG = True
