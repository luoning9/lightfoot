import gradio as gr

import myutils
from chat_index import ChatIndex

global_conf = myutils.initialize_config()
INDEX_ID = global_conf.get('default', 'INDEX_ID')
assert INDEX_ID, "no index id set!"
# ======= end of init ===============


with gr.Blocks() as app:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    # init chat index
    chat_index = ChatIndex(INDEX_ID)


    def get_model_reply(query, context):
        engine = chat_index.chat_engine()
        response = engine.chat(query)

        context.append((query, str(response)))
        # print(f"OUT: {context}")
        return "", context


    msg.submit(get_model_reply, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    app.launch()
