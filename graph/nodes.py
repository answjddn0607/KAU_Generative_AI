from .state import AgentState
from prompts.system_prompt import SYSTEM_PROMPT
import json
from openai import OpenAI
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage


def agent_node(state: AgentState) -> AgentState:
    """Agent 노드"""
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    from tools.tool_registry import ToolRegistry, register_all_tools
    
    registry = ToolRegistry()
    register_all_tools(registry)
    
    tools = registry.list_openai_tools()
    messages = list(state["messages"])
    
    formatted_messages = []
    has_system = any(isinstance(m, SystemMessage) for m in messages)
    
    if not has_system:
        formatted_messages.append({"role": "system", "content": SYSTEM_PROMPT})
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            msg_dict = {"role": "assistant", "content": msg.content or ""}
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                openai_tool_calls = [
                    {
                        "id": tc.get("id"),
                        "type": "function",
                        "function": {
                            "name": tc.get("name"),
                            "arguments": json.dumps(tc.get("args", {}))
                        }
                    }
                    for tc in msg.tool_calls
                ]
                msg_dict["tool_calls"] = openai_tool_calls
            formatted_messages.append(msg_dict)
        elif isinstance(msg, SystemMessage):
            formatted_messages.append({"role": "system", "content": msg.content})
        elif isinstance(msg, ToolMessage):
            formatted_messages.append({
                "role": "tool",
                "content": msg.content,
                "tool_call_id": msg.tool_call_id
            })
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=formatted_messages,
        tools=tools if tools else None,
        tool_choice="auto",
        parallel_tool_calls=False
    )
    
    msg = response.choices[0].message
    
    if msg.tool_calls:
        tool_calls_for_langchain = [
            {
                "name": tc.function.name,
                "args": json.loads(tc.function.arguments),
                "id": tc.id,
                "type": "tool_call"
            }
            for tc in msg.tool_calls
        ]
        
        ai_msg = AIMessage(
            content=msg.content or "",
            tool_calls=tool_calls_for_langchain
        )
        return {
            "messages": [ai_msg],
            "tool_result": json.dumps([tc.model_dump() for tc in msg.tool_calls]),
            "iteration": state["iteration"] + 1
        }
    
    return {
        "messages": [AIMessage(content=msg.content or "")],
        "tool_result": None,
        "iteration": state["iteration"] + 1
    }


def tools_node(state: AgentState) -> AgentState:
    """Tool 실행 노드"""
    
    from tools.tool_registry import ToolRegistry, register_all_tools
    
    registry = ToolRegistry()
    register_all_tools(registry)

    tool_calls = json.loads(state["tool_result"])

    if not isinstance(tool_calls, list):
        tool_calls = [tool_calls]
    
    observations = []

    for tool_call in tool_calls:
        name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])
        
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
        
        observation = ToolMessage(
            content=output,
            tool_call_id=tool_call["id"]
        )
        observations.append(observation)
    
    return {
        "messages": observations,
        "tool_result": None
    }