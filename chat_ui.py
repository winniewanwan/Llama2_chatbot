import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_recipes.inference.chat_utils import format_tokens
import gradio as gr


# MODEL_ID = '/home/null/D/Workspace/Code/python/GitHub/sonodasong/demo/llm/Llama-2-7b-chat-hf'
MODEL_ID = '/home/null/D/Workspace/Code/python/GitHub/sonodasong/demo/llm/Llama-2-13b-chat-hf'

SYSTEM = '''
You are a virtual human resource assistant.
You need to generate freelancer job postings based on company information and company requirement.
The job postings should job description, price offer and task duration.
'''


class ChatUi:

    MAX_CHAT = 384
    HEIGHT = 1000

    def __init__(self, model_id):
        self.get_tokenizer(model_id)
        self.get_model(model_id)
        self.get_ui()

    def get_tokenizer(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    def get_model(self, model_id):
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', load_in_8bit=True)
        self.device = self.model.device

    def get_ui(self):
        self.ui = gr.ChatInterface(
            self.response,
            chatbot=gr.Chatbot(height=self.HEIGHT)
        )

    def fill(self, role, content):
        return {'role': role, 'content': content}

    def fill_system(self, content):
        return self.fill('system', content)

    def fill_user(self, content):
        return self.fill('user', content)

    def fill_assistant(self, content):
        return self.fill('assistant', content)

    def format(self, chats):
        return format_tokens([chats], self.tokenizer)[0]

    def generate(self, tokens):
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=torch.as_tensor(tokens).long().to(self.device).unsqueeze(dim=0),
                max_new_tokens=self.MAX_CHAT,
                do_sample=True,
                top_p=1.0,
                temperature=1.0,
                use_cache=True,
                top_k=50,
                repetition_penalty=1.0,
                length_penalty=1
            )
            return outputs[0].tolist()

    def convert(self, content, history):
        chats = []
        if self.system is not None:
            chats.append(self.fill_system(self.system))
        for user, assistant in history:
            chats.append(self.fill_user(user))
            chats.append(self.fill_assistant(assistant))
        chats.append(self.fill_user(content))
        return chats

    def response(self, user, history):
        chats = self.convert(user, history)
        tokens = self.format(chats)
        n = len(tokens)
        tokens = self.generate(tokens)
        return self.tokenizer.decode(tokens[n:], skip_special_tokens=True)

    def run(self, system=None):
        self.system = system
        self.ui.launch(server_name='0.0.0.0')


if __name__ == '__main__':
    chat_ui = ChatUi(MODEL_ID)
    chat_ui.run(SYSTEM)
