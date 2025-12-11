from .graph import create_graph
from .state import AgentState
import json


def run_with_interrupt(user_input: str, session_id: str = "default"):
    """Interrupt 모드"""
    
    graph = create_graph()
    config = {"configurable": {"thread_id": session_id}}
    
    initial_state: AgentState = {
        "messages": [{"role": "user", "content": user_input}],
        "tool_result": None,
        "iteration": 0,
        "max_iterations": 5
    }
    
    print("\n=== 1. Agent 실행 중... ===")
    graph.invoke(initial_state, config=config)
    
    snapshot = graph.get_state(config)
    print("\n=== 2. Tool 실행 대기 중 ===")
    
    if snapshot.values.get("tool_result"):
        tool_info = json.loads(snapshot.values["tool_result"])
        print(f"실행 예정 Tool:")
        print(f"  - {tool_info['function']['name']}")
        print(f"  - Args: {tool_info['function']['arguments']}")
        
        confirm = input("\n실행하시겠습니까? (y/n): ")
        
        if confirm.lower() != 'y':
            print("❌ 사용자가 취소했습니다.")
            return None
    
    print("\n=== 3. 실행 재개 ===")
    result = graph.invoke(None, config=config)
    
    final_msg = result["messages"][-1]
    return final_msg.get("content", "")


def run_with_stream(user_input: str, session_id: str = "default"):
    """Stream 모드"""
    
    graph = create_graph()
    config = {"configurable": {"thread_id": session_id}}
    
    initial_state: AgentState = {
        "messages": [{"role": "user", "content": user_input}],
        "tool_result": None,
        "iteration": 0,
        "max_iterations": 5
    }
    
    print("🚀 Agent 시작...\n")
    
    for event in graph.stream(initial_state, config, stream_mode="values"):
        iteration = event.get("iteration", 0)
        
        print(f"[Iteration {iteration}]")
        
        if event.get("messages"):
            last_msg = event["messages"][-1]
            role = last_msg.get("role", "unknown")
            
            if role == "assistant":
                if last_msg.get("tool_calls"):
                    print(f"  🔧 Tool Call:")
                    for tc in last_msg.get("tool_calls", []):
                        print(f"    - {tc['function']['name']}")
                elif last_msg.get("content"):
                    preview = last_msg["content"][:100]
                    print(f"  💬 Response: {preview}...")
            
            elif role == "tool":
                print(f"  📊 Tool Result")
        
        print("-" * 50)
    
    final_msg = event["messages"][-1]
    return final_msg.get("content", "")
