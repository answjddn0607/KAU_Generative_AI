from .state import AgentState
from prompts.system_prompt import SYSTEM_PROMPT
import json
from openai import OpenAI

client = OpenAI()


def agent_node(state: AgentState) -> AgentState:
    """Agent 노드"""
    
    from tools.tool_registry import ToolRegistry, register_all_tools, list_openai_tools
    
    registry = ToolRegistry()
    register_all_tools(registry)
    
    tools = list_openai_tools(registry)
    messages = list(state["messages"])
    
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools if tools else None,
        tool_choice="auto"
    )
    
    msg = response.choices[0].message
    
    if msg.tool_calls:
        tool_call = msg.tool_calls[0]
        
        return {
            "messages": [msg.to_dict()],
            "tool_result": json.dumps(tool_call.to_dict()),
            "iteration": state["iteration"] + 1
        }
    
    return {
        "messages": [msg.to_dict()],
        "tool_result": None,
        "iteration": state["iteration"] + 1
    }


def tools_node(state: AgentState) -> AgentState:
    """Tool 실행 노드"""
    
    from tools.tool_registry import ToolRegistry, register_all_tools
    
    registry = ToolRegistry()
    register_all_tools(registry)
    
    json_load_tool = json.loads(state["tool_result"])
    name = json_load_tool["function"]["name"]
    args = json.loads(json_load_tool["function"]["arguments"])
    
    print(f"[Executing Tool] {name} with args: {args}")
    
    try:
        result = registry.call(name, args)
        output = json.dumps(result, ensure_ascii=False)
    except Exception as e:
        output = json.dumps({
            "error": str(e),
            "tool": name
        }, ensure_ascii=False)
    
    print(f"[Tool 결과] {output[:200]}...")
    
    observation = {
        "role": "tool",
        "content": output,
        "tool_call_id": json_load_tool["id"],
    }
    
    return {
        "messages": [observation],
        "tool_result": None
    }