import json
from openai import OpenAI
from prompts.memory_prompt import MEMORY_EXTRACTOR_PROMPT

client = OpenAI()

# 대화 끝나면 자동으로 저장 여부 판단(Reflection)
def extract_and_save_memory(question: str, answer: str):
    from tools.tool_registry import ToolRegistry, register_all_tools
    
    snippet = f"User: {question}\nAssistant: {answer}"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": MEMORY_EXTRACTOR_PROMPT},
            {"role": "user", "content": snippet}
        ],
        temperature=0
    )
    
    try:
        decision = json.loads(response.choices[0].message.content)
        
        if decision.get("should_write_memory"):
            registry = ToolRegistry()
            register_all_tools(registry)
            
            registry.call("memory_write", {
                "content": decision["content"],
                "memory_type": decision.get("memory_type", "episodic"),
                "importance": decision.get("importance", 3),
                "tags": decision.get("tags", [])
            })
            print(f"[Memory Saved] {decision['content'][:50]}...")
    except Exception as e:
        print(f"[Reflection Error] {e}")