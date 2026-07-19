import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "pinkelephantlimited/pink-elephant-llm-1.3b"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def generate(prompt, max_length=512, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def chat(message, history):
    history = history or []
    prompt = message
    if history:
        context = "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in history])
        prompt = f"{context}\nUser: {message}\nAssistant:"
    response = generate(prompt)
    return response


css = """
footer {display: none !important;}
"""

with gr.Blocks(css=css, title="Pink Elephant LLM 1.3B") as demo:
    gr.Markdown("# 🐘 Pink Elephant LLM 1.3B")
    gr.Markdown("A conversational demo of the **Pink Elephant LLM 1.3B** model. Type a message below and the model will respond.")
    chatbot = gr.Chatbot(label="Conversation", height=400)
    msg = gr.Textbox(label="Your message", placeholder="Say something to the Pink Elephant...", scale=4)
    with gr.Row():
        submit = gr.Button("Send", variant="primary", scale=1)
        clear = gr.ClearButton([msg, chatbot], value="Clear")

    def respond(message, chat_history):
        bot_msg = chat(message, chat_history)
        chat_history.append((message, bot_msg))
        return "", chat_history

    submit.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()
