import os
import gradio as gr
import openai

import myutils

global_conf = myutils.initialize_config()

assert os.getenv("OPENAI_API_KEY") is not None, "please set openai key!"
assert os.getenv("all_proxy") is not None, "please set proxy!"

openai.api_key = os.getenv("OPENAI_API_KEY")

LLM_MODEL = global_conf.get('query','model')
if LLM_MODEL is None:
    LLM_MODEL = 'gpt-3.5-turbo'
# ======= end of init ===============
chat_history = [
    {"role": "system", "content": "You are a helpful and kind AI Assistant."},
]


def get_model_reply(query, context):
    chat_history.append({"role": "user", "content": query})
    # given the most recent context (4096 characters)
    # continue the text up to 2048 tokens ~ 8192 characters
    response = openai.ChatCompletion.create(
        model=LLM_MODEL,  # one of the most capable models available
        messages=chat_history,
    )
    # append response to context
    reply = response.choices[0].message.content
    chat_history.append({"role": "assistant", "content": reply})
    # list of (user, bot) responses. We will use this format later
    messages = [(u['content'], b['content']) for u, b in zip(chat_history[1::2], chat_history[2::2])]
    # print(f"OUT: {messages}")
    return "", messages


with gr.Blocks() as app:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    msg.submit(get_model_reply, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    app.launch()
