from fastapi import FastAPI
import gradio as gr
import uvicorn
import uuid
import time 
from graph.runner import run_with_stream

# FastAPI 앱 생성
app = FastAPI()

with gr.Blocks(title="Transporter", fill_height=True) as demo:
    
    # 사이드바: 로그창
    with gr.Sidebar(label="Agent Status"):
        log_view = gr.Markdown("")

    gr.Markdown(
        """
        # Transporter
        ### KAU Generative AI
        """
    )
    
    chatbot = gr.Chatbot(label="대화창", height=450)
    msg = gr.Textbox(label="질문 입력")
    session_id_state = gr.State(lambda: str(uuid.uuid4()))
    log_history_state = gr.State(value="") 

    def respond(user_message, history, session_id, log_accumulated):
        if not user_message:
            return "", history, "내용을 입력해주세요.", log_accumulated

        if history is None:
            history = []
        
        history.append({"role": "user", "content": user_message})

        # 답변 출력 대기
        history.append({"role": "assistant", "content": "..."}) 
        
        prefix = ""
        if log_accumulated:
            prefix = log_accumulated + "\n\n---\n\n"
            
        current_header = f"### 🔎 질문: {user_message}\n"

        yield "", history, prefix + current_header + "에이전트가 작업을 시작했습니다...", log_accumulated
        
        for output in run_with_stream(user_message, session_id=session_id):
            if "**최종 답변:**" in output:
                parts = output.split("**최종 답변:**")
                logs = parts[0].strip()
                full_answer = parts[1].strip()
                
                new_log_entry = prefix + current_header + logs
                
                # 답변을 출력하기 직전에 "..."을 지움
                history[-1]['content'] = ""
                
                for char in full_answer:
                    history[-1]['content'] += char
                    yield "", history, new_log_entry, new_log_entry
                    time.sleep(0.005) 

            else:
                # 로그만 업데이트 (화면엔 여전히 "..." 표시됨)
                current_view = prefix + current_header + output
                yield "", history, current_view, log_accumulated

    msg.submit(
        respond, 
        [msg, chatbot, session_id_state, log_history_state], 
        [msg, chatbot, log_view, log_history_state]
    )

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)