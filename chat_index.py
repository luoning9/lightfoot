import logging

from llama_index import load_index_from_storage, ServiceContext
from llama_index.indices.base import BaseIndex
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.llms import OpenAI

import myutils


class ChatIndex:
    llamaIndex: BaseIndex
    queryEngine: BaseQueryEngine

    def __init__(self, index_id: str):
        conf = myutils.global_light.config
        top_k = int(conf.get('query', 'similarity_top_k'))
        model = conf.get('query', 'model')

        storage_context = myutils.get_mongo_storage()
        index = load_index_from_storage(storage_context, index_id=index_id)
        llm = OpenAI(model=model)

        service_context = ServiceContext.from_defaults(llm=llm)
        self.llamaIndex = index
        self.queryEngine = index.as_query_engine(service_context=service_context,
                                                 similarity_top_k=top_k,
                                                 streaming=True)
        logging.info("query engine ready for '%s'.", index_id)


chat_indices_cache = {}


def load_index(index_id: str):
    index = chat_indices_cache.get(index_id)
    if index is None:
        index = ChatIndex(index_id)
        chat_indices_cache[index_id] = index
    return index
