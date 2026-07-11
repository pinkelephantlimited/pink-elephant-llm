from typing import Optional

from .engine import InferenceEngine


def generate_text(
    engine: InferenceEngine,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
) -> str:
    return engine.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )


def chat(
    engine: InferenceEngine,
    system_prompt: str = "You are Pink Elephant, a helpful AI assistant.",
    max_turns: int = 10,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
):
    conversation = f"<|system|>{system_prompt}</s>"

    print("Pink Elephant:1B - Interactive Chat")
    print("Type 'exit' to quit\n")

    for _ in range(max_turns):
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            break

        conversation += f"<|user|>{user_input}</s><|assistant|>"
        response = engine.generate(
            prompt=conversation,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        response = response.replace(conversation, "").strip()
        print(f"Pink Elephant: {response}\n")
        conversation += f"{response}</s>"
