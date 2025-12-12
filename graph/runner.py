from .graph import create_graph
from .state import AgentState
import json
from langchain_core.messages import HumanMessage


def run_with_interrupt(user_input: str, session_id: str = "default"):
    """Interrupt 모드"""
    
    graph = create_graph(interrupt=True)
    config = {"configurable": {"thread_id": session_id}}
    
    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_input)],
        "tool_result": None,
        "iteration": 0,
        "max_iterations": 5
    }
    
    print("\n=== 1. Agent 실행 중 (interrupt_before=['tools']) ===")
    for event in graph.stream(initial_state, config, stream_mode="values"):
        if event.get("iteration", 0) > 0:
            print(f"[Iteration {event['iteration']}] Agent 노드 완료")
    
    snapshot = graph.get_state(config)
    
    print(f"\n=== 2. Tool 실행 대기 중 (next: {snapshot.next}) ===")
    if snapshot.values.get("tool_result"):
        tool_calls = json.loads(snapshot.values["tool_result"])
        
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]
        
        print(f"실행 예정 Tool ({len(tool_calls)}개):")
        for i, tc in enumerate(tool_calls, 1):
            print(f"  {i}. {tc['function']['name']}")
            print(f"     Args: {tc['function']['arguments']}")
        
        confirm = input("\n실행하시겠습니까? (y/n): ")
        
        if confirm.lower() != 'y':
            print("❌ 사용자가 취소했습니다.")
            return None
    
    print("\n=== 3. 실행 재개 ===")
    for event in graph.stream(None, config, stream_mode="values"):
        iteration = event.get("iteration", 0)
        
        if event.get("messages"):
            last_msg = event["messages"][-1]
            
            # 메시지 객체 처리
            if hasattr(last_msg, '__class__'):
                msg_type = last_msg.__class__.__name__
                if msg_type == "ToolMessage":
                    print(f"[Iteration {iteration}] Tool 실행 완료")
                elif msg_type == "AIMessage" and (not hasattr(last_msg, 'tool_calls') or not last_msg.tool_calls):
                    print(f"[Iteration {iteration}] 최종 답변 생성")

    final_state = graph.get_state(config)
    final_msg = final_state.values["messages"][-1]
    
    return final_msg.content if hasattr(final_msg, 'content') else ""


def run_with_stream(user_input: str, session_id: str = "default"):
    """Stream 모드"""
    
    graph = create_graph(interrupt=False)
    config = {"configurable": {"thread_id": session_id}}
    
    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_input)],
        "tool_result": None,
        "iteration": 0,
        "max_iterations": 5
    }
    
    print("🚀 Agent 시작 (Stream Mode)...\n")
    
    for event in graph.stream(initial_state, config, stream_mode="updates"):
        for node_name, node_output in event.items():
            iteration = node_output.get("iteration", 0)
            
            print(f"[{node_name.upper()}] Iteration {iteration}")
            
            if node_name == "agent":
                messages = node_output.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    
                    # AIMessage 객체 처리
                    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                        tool_calls = last_msg.tool_calls
                        print(f"  🔧 Tool Call ({len(tool_calls)}개):")
                        for tc in tool_calls:
                            # LangGraph 형식의 tool_call
                            print(f"    - {tc['name']}")
                    elif hasattr(last_msg, 'content') and last_msg.content:
                        preview = last_msg.content[:100]
                        print(f"  💬 Response: {preview}...")
            
            elif node_name == "tools":
                messages = node_output.get("messages", [])
                print(f"  📊 Tool 결과: {len(messages)}개")
            
            print("-" * 50)
    
    final_state = graph.get_state(config)
    final_msg = final_state.values["messages"][-1]
    
    # 메시지 객체에서 content 추출
    return final_msg.content if hasattr(final_msg, 'content') else ""
