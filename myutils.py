import logging

import pymongo
from llama_index import StorageContext
from llama_index.storage.docstore import MongoDocumentStore
from llama_index.storage.index_store import MongoIndexStore
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

import config


def clear_mongo_stores() -> None:
    config_parser = config.initialize_config()
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
    config_parser = config.initialize_config()
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
