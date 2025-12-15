from .graph import create_graph
from .state import AgentState
from memory.reflection import extract_and_save_memory
from langchain_core.messages import HumanMessage


def run_with_stream(user_input: str, session_id: str = "default"):
    """Stream + Interrupt 모드"""
    
    graph = create_graph(interrupt=True)
    config = {"configurable": {"thread_id": session_id}}
    
    # 기존 상태 확인 (재개인지 체크)
    snapshot = graph.get_state(config)
    
    if snapshot.next:  # interrupt 상태면 재개
        print(f"[RESUME] 재개 - next: {snapshot.next}")
        graph.update_state(config, {"query": user_input})
        initial_state = None
    else:
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "tool_result": None,
            "iteration": 0,
            "max_iterations": 5,
            "query": "",
            "status": "",
            "rag_result": None,
            "api_result": None,
            "google_result": None,
            "target_paper": None,
            "related_papers": None,
            "user_interests": None,
            "recommendations": None,
            "final_result": None
        }
    
    print("🚀 Agent 시작 (Stream Mode)...\n")
    logs = "🚀 **Agent 시작** (LangGraph Running...)\n"
    yield logs
    
    for event in graph.stream(initial_state, config, stream_mode="updates"):
        for node_name, node_output in event.items():
            
            # 1. 에이전트가 말하거나 도구를 호출했을 때
            if node_name == "agent":
                messages = node_output.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    
                    # 도구 호출이 있는 경우에만
                    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                        
                        logs += f"\n\n🛠️ **도구 호출** ({len(last_msg.tool_calls)}개):\n"
                        for tc in last_msg.tool_calls:
                            func_name = tc['name']
                            func_args = tc['args']
                            logs += f"- ⚙️ **Running:** `{func_name}`\n"
                            logs += f"  - 📥 **Input:** `{str(func_args)}`\n"
                        logs += "\n"
                        yield logs
                    
                    # 최종 답변 생성
                    else:
                        pass

            # 2. Tools 실행을 마치고 결과를 뱉었을 때
            elif node_name == "tools": 
                messages = node_output.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    result_preview = last_msg.content[:200]
                    
                    logs += f"\n\n✅ **도구 실행 완료!**\n"
                    logs += f"> 📤 **Output:** {result_preview}...\n"
                    yield logs

            # 3. 그 외 커스텀 노드 
            else:
                logs += f"\n\n🔄 **작업 중:** `{node_name}` 단계 수행 중...\n"
                yield logs
    
    final_state = graph.get_state(config)
    final_msg = final_state.values["messages"][-1]
    answer = final_msg.content if hasattr(final_msg, 'content') else "답변 생성 실패"

    extract_and_save_memory(user_input, answer)
    
    logs += "\n\n✅ **작업이 완료되었습니다.**"
    yield logs + f"\n\n**최종 답변:**\n\n{answer}"