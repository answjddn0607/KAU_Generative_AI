import os
from dotenv import load_dotenv
from pathlib import Path
import traceback

env_path = Path(__file__).parent / "key.env"
load_dotenv(dotenv_path=env_path)

from graph.runner import run_with_interrupt, run_with_stream
from tools.tool_registry import ToolRegistry, register_all_tools

def test_stream():
    """Stream 모드 기본 테스트"""
    print("=" * 70)
    print("🧪 TEST 1: Stream 모드 (간단한 질문)")
    print("=" * 70)
    
    result = run_with_stream(
        "2 + 3은 얼마야?",
        session_id="test-stream-1"
    )
    
    print("\n✅ 최종 결과:")
    print(result)
    print("\n" + "=" * 70 + "\n")


def test_interrupt():
    """Interrupt 모드 테스트"""
    print("=" * 70)
    print("🧪 TEST 2: Interrupt 모드 (Tool 호출)")
    print("=" * 70)
    
    result = run_with_interrupt(
        "구글에서 'LangGraph'를 검색해줘",
        session_id="test-interrupt-1"
    )
    
    if result:
        print("\n✅ 최종 결과:")
        print(result[:200] + "..." if len(result) > 200 else result)
    else:
        print("\n❌ 사용자가 취소했습니다.")
    
    print("\n" + "=" * 70 + "\n")


def test_tool_registry():
    """Tool Registry 확인"""
    print("=" * 70)
    print("🧪 TEST 0: Tool Registry 확인")
    print("=" * 70)
    
    registry = ToolRegistry()
    register_all_tools(registry)
    
    print(f"\n등록된 Tool ({len(registry._tools)}개):")
    for name, spec in registry._tools.items():
        print(f"  - {name}: {spec.description[:50]}...")
    
    print("\n" + "=" * 70 + "\n")


def main():
    """메인 테스트 함수"""
    
    print("\n🚀 LangGraph 테스트 시작\n")
    
    # 0. Tool Registry 확인
    test_tool_registry()
    
    # 1. Stream 모드 테스트
    try:
        test_stream()
    except Exception as e:
        print(f"❌ Stream 테스트 실패: {e}")
        print("\n상세 에러:")
        traceback.print_exc()
        print()
    
    # 2. Interrupt 모드 테스트
    try:
        test_interrupt()
    except Exception as e:
        print(f"❌ Interrupt 테스트 실패: {e}")
        print("\n상세 에러:")
        traceback.print_exc()
        print()
    
    print("✅ 모든 테스트 완료!")


if __name__ == "__main__":
    main()
