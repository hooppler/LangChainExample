import gradio as gr


def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)


# demo = gr.Interface(
#     fn=greet,
#     inputs=["text", "slider"],
#     outputs=["text"],
# )
#
#
# tts_examples = [
#     "I love learning machine learning",
#     "How do you do?",
# ]
#
# tts_demo = gr.load(
#     "huggingface/facebook/fastspeech2-en-ljspeech",
#     title=None,
#     examples=tts_examples,
#     description="Give me something to say!",
# )
#
# stt_demo = gr.load(
#     "huggingface/facebook/wav2vec2-base-960h",
#     title=None,
#     inputs=gr.Microphone(type="filepath"),
#     description="Let me try to guess what you're saying!",
# )


def update(name):
    return f"Welcome to Gradio, {name}!"


markdown = gr.Markdown("# ChatWithYourData_Bot")

with gr.Blocks() as conversation_tab:
    with gr.Row():
        inp = gr.Textbox(placeholder="What is your name?")
        btn = gr.Button("Send question")
    with gr.Row():
        out = gr.Textbox()
        btn.click(fn=update, inputs=inp, outputs=out)

with gr.Blocks() as database_tab:
    with gr.Column():
        gr.Markdown("Last question to db:")
        inp = gr.Textbox(placeholder="What is your name?")
    with gr.Row():
        out = gr.TextArea(placeholder="Some text")


with gr.Blocks() as chat_history_tab:
    gr.Markdown("Start typing below and then click **Run** to see the output.")
    with gr.Column():
        inp = gr.Textbox(placeholder="What is your name?")
        out = gr.Textbox()
    btn = gr.Button("Run")
    btn.click(fn=update, inputs=inp, outputs=out)

with gr.Blocks() as configure_tab:
    gr.Markdown("Start typing below and then click **Run** to see the output.")
    with gr.Column():
        inp = gr.Textbox(placeholder="What is your name?")
        out = gr.Textbox()
    btn = gr.Button("Run")
    btn.click(fn=update, inputs=inp, outputs=out)


demo = gr.TabbedInterface(
    [conversation_tab, database_tab, chat_history_tab, configure_tab],
    ["Conversation", "Database", "Chat History", "Configure"])


# dashboard = pn.Column(
#     pn.Row(pn.pane.Markdown('# ChatWithYourData_Bot')),
#     pn.Tabs(('Conversation', tab1), ('Database', tab2), ('Chat History', tab3),('Configure', tab4))



demo.launch()

