# 1단계 · LLM 호출해 보기

## 목표

- 파이썬 코드로 LLM에게 질문하고 답을 받는다.
- **시스템 프롬프트**로 LLM의 역할과 말투를 바꿀 수 있다는 걸 확인한다.
- LLM은 **이전 대화를 스스로 기억하지 못한다**는 걸 확인한다.

## 실행 순서

```bash
python 01_llm_basics/step1_hello.py
python 01_llm_basics/step2_system_prompt.py
python 01_llm_basics/step3_memory.py
```

> 첫 실행은 모델을 메모리에 올리느라 10~30초쯤 걸릴 수 있습니다. 두 번째부터 빨라집니다.

## 핵심 개념

### LLM 호출 = 메시지 목록을 보내고, 메시지 하나를 받는 것

```
[보내는 것]                          [받는 것]
SystemMessage  "너는 ~ 역할이야"  ─┐
HumanMessage   "질문"            ─┴─▶  AIMessage "답변"
```

- `SystemMessage`: LLM의 역할·규칙. 사용자에게는 안 보이는 "업무 지침서".
- `HumanMessage`: 사용자의 질문.
- `AIMessage`: LLM의 답.

### LLM은 기억이 없다

LLM은 호출할 때마다 **받은 메시지 목록만** 봅니다.
ChatGPT 같은 서비스가 대화를 기억하는 것처럼 보이는 이유는,
프로그램이 지금까지의 대화를 **매번 통째로 다시 보내기** 때문입니다.
`step3_memory.py`에서 직접 확인해 보세요.

## 해볼 것

1. `step1_hello.py`의 질문을 바꿔 보세요.
2. `step2_system_prompt.py`에 나만의 역할(예: "신입 사원에게 설명하는 선배")을 추가해 보세요.
3. `step3_memory.py`에서 `history`에 대화를 더 쌓아 보세요.
