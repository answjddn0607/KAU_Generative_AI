from fastapi import FastAPI
import gradio as gr
import uvicorn
import uuid
from graph.runner import run_with_stream

# FastAPI 앱 생성
app = FastAPI()

# Gradio 로직
def respond(message, history, session_id):
    for output in run_with_stream(message, session_id=session_id):
        yield output

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Transpoter
        ### KAU Generative AI
        """
    )
    session_id_state = gr.State(value=lambda: str(uuid.uuid4()))
    chat_interface = gr.ChatInterface(
        fn=respond,
        additional_inputs=[session_id_state]
    )

# mount
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)