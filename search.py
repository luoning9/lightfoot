import myutils
from chat_index import ChatIndex

global_conf = myutils.initialize_config()
INDEX_ID = global_conf.get('default', 'INDEX_ID')
assert INDEX_ID, "no index id set!"
# ======= end of init ===============

query = ChatIndex(INDEX_ID).chat_engine()
response = query.chat("超分辨率图像如何实现重建？")
# response = query.chat("ping")
print(response)
