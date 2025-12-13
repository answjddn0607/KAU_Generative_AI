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
            # interrupt 처리
            if node_name == "__interrupt__":
                print("[INTERRUPT] 감지!")
                snap = graph.get_state(config)
                if snap.next and "pa_ask_user" in snap.next:
                    query = snap.values.get("query", "")
                    answer = f"'{query}' 관련 논문을 찾지 못했습니다. 정확한 논문 제목을 입력해주세요."
                    yield logs + f"\n\n⏸️ **입력 대기:**\n\n{answer}"
                    return
                continue
            
            if not isinstance(node_output, dict):
                continue
            
            iteration = node_output.get("iteration", 0)
            print(f"[{node_name.upper()}] Iteration {iteration}")
            
            if node_name == "agent":
                messages = node_output.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                        tool_calls = last_msg.tool_calls
                        print(f"  🔧 Tool Call ({len(tool_calls)}개):")
                        logs += f"🛠️ **Tool Call** ({len(tool_calls)}개):\n"
                        for tc in tool_calls:
                            print(f"    - {tc['name']}")
                            logs += f"  - `{tc['name']}`\n"
                        yield logs
                    elif hasattr(last_msg, 'content') and last_msg.content:
                        print(f"  💬 Response: {last_msg.content[:100]}...")
    
    final_state = graph.get_state(config)
    final_msg = final_state.values["messages"][-1]
    answer = final_msg.content if hasattr(final_msg, 'content') else "답변 생성 실패"

    extract_and_save_memory(user_input, answer)
    
    yield logs + f"\n\n✅ **최종 답변:**\n\n{answer}"