from fastapi import FastAPI
import gradio as gr
import uvicorn
import uuid
from graph.runner import run_with_stream

# FastAPI 앱 생성
app = FastAPI()

with gr.Blocks(title="Transpoter", fill_height=True) as demo:
    
    # 사이드바: 로그창
    with gr.Sidebar(label="Agent Status"):
        gr.Markdown("### 실시간 로그")
        log_view = gr.Markdown("대기 중...")

    gr.Markdown(
        """
        # Transporter
        ### KAU Generative AI
        """
    )
    
    chatbot = gr.Chatbot(label="대화창", height=450)
    
    msg = gr.Textbox(label="질문 입력")
    
    session_id_state = gr.State(lambda: str(uuid.uuid4()))

    def respond(user_message, history, session_id):
        if not user_message:
            return "", history, "내용을 입력해주세요."

        if history is None:
            history = []
        
        # 사용자 질문 추가
        history.append({"role": "user", "content": user_message})
        
        # AI 답변 자리 만들기 (빈 내용으로 미리 추가)
        history.append({"role": "assistant", "content": "..."})
        
        yield "", history, "에이전트가 작업을 시작했습니다..."
        
        for output in run_with_stream(user_message, session_id=session_id):
            if "**최종 답변:**" in output:
                parts = output.split("**최종 답변:**")
                logs = parts[0].strip()
                answer = parts[1].strip()
                
                # 마지막 AI 답변 내용을 실제 정답으로 업데이트
                history[-1]['content'] = answer
                
                yield "", history, logs
            else:
                # [진행 중] 로그창만 업데이트
                yield "", history, output

    msg.submit(respond, [msg, chatbot, session_id_state], [msg, chatbot, log_view])

# Mount
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)