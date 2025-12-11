from typing import Dict, Any, List
from pydantic import ValidationError

from .tool_definitions import (
    ToolSpec,
    # Google Search
    GoogleSearchInput,
    google_search_handler,
    # Memory
    MemoryWriteInput,
    MemoryReadInput,
    memory_write_handler,
    memory_read_handler,
    # RAG
    RAGIndexInput,
    RAGSearchInput,
    rag_index_handler,
    rag_search_handler,
)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register_tool(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"이미 등록된 툴입니다: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"존재하지 않는 툴입니다: {name}")
        return self._tools[name]

    def call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM이 준 args(dict)를 Pydantic input_model로 검증 후
        handler로 넘겨 실행한다.
        """
        spec = self.get(name)

        try:
            input_obj = spec.input_model.model_validate(args)
        except ValidationError as e:
            return {
                "error": "INVALID_TOOL_ARGS",
                "detail": str(e),
            }
        return spec.handler(input_obj)
    
    def list_openai_tools(self) -> List[Dict[str, Any]]:
        return [self.as_openai_tool_spec(spec) for spec in self._tools.values()]

    # JSONㅎ 형태로 변환
    def as_openai_tool_spec(self, spec: ToolSpec) -> Dict[str, Any]:
        schema = spec.input_model.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": schema,
            },
        }


def register_all_tools(registry: ToolRegistry) -> None:
    """
    모든 툴 등록
    새로운 툴 추가 시 여기에 추가
    """
    # 1. Google Search
    registry.register_tool(
        ToolSpec(
            name="google_search",
            description="Google CSE로 웹 검색을 수행하는 툴",
            input_model=GoogleSearchInput,
            handler=google_search_handler,
        )
    )
    
    # 2. Memory Write
    registry.register_tool(
        ToolSpec(
            name="memory_write",
            description="중요한 정보를 메모리에 저장합니다. 나중에 참조할 내용을 기억할 때 사용합니다.",
            input_model=MemoryWriteInput,
            handler=memory_write_handler,
        )
    )
    
    # 3. Memory Read
    registry.register_tool(
        ToolSpec(
            name="memory_read",
            description="저장된 메모리에서 관련 정보를 검색합니다. 이전에 저장한 내용을 찾을 때 사용합니다.",
            input_model=MemoryReadInput,
            handler=memory_read_handler,
        )
    )
    
    # 4. RAG Index
    registry.register_tool(
        ToolSpec(
            name="rag_index",
            description="논문 정보(제목, 초록, 저자)를 인덱싱합니다. PDF에서 추출한 논문을 저장할 때 사용합니다.",
            input_model=RAGIndexInput,
            handler=rag_index_handler,
        )
    )
    
    # 5. RAG Search
    registry.register_tool(
        ToolSpec(
            name="rag_search",
            description="인덱싱된 논문 초록에서 유사한 논문을 검색합니다. 관련 연구를 찾을 때 사용합니다.",
            input_model=RAGSearchInput,
            handler=rag_search_handler,
        )
    )
