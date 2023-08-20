import logging

from llama_index import load_index_from_storage, ServiceContext
from llama_index.chat_engine.types import ChatMode, BaseChatEngine
from llama_index.indices.base import BaseIndex
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.llms import OpenAI

import myutils


class ChatIndex:
    llamaIndex: BaseIndex
    chatEngine: BaseChatEngine = None
    queryEngine: BaseQueryEngine = None

    def __init__(self, index_id: str):
        storage_context = myutils.get_mongo_storage()
        self.llamaIndex = load_index_from_storage(storage_context, index_id=index_id)

    def query_engine(self) -> BaseQueryEngine:
        if self.queryEngine is None:
            conf = myutils.global_light.config
            top_k = int(conf.get('query', 'similarity_top_k'))
            model = conf.get('query', 'model')
            llm = OpenAI(model=model)
            service_context = ServiceContext.from_defaults(llm=llm)
            self.queryEngine = self.llamaIndex.as_query_engine(service_context=service_context,
                                                               similarity_top_k=top_k,
                                                               streaming=False)
            logging.info("query engine ready for '%s'.", self.llamaIndex.index_id)

        return self.queryEngine

    def chat_engine(self) -> BaseChatEngine:
        if self.chatEngine is None:
            conf = myutils.global_light.config
            top_k = int(conf.get('query', 'similarity_top_k'))
            model = conf.get('query', 'model')
            llm = OpenAI(model=model)
            service_context = ServiceContext.from_defaults(llm=llm)
            self.chatEngine = self.llamaIndex.as_chat_engine(chat_mode=ChatMode.BEST,
                                                             service_context=service_context,
                                                             similarity_top_k=top_k,
                                                             verbose=True)
            logging.info("chat engine created for '%s'.",  self.llamaIndex.index_id)

        return self.chatEngine


