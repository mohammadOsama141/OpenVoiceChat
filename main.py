#from openvoicechat.tts.tts_xtts import Mouth_xtts
# from openvoicechat.tts.tts_hf import Mouth_hf
# from openvoicechat.tts.tts_hf import Mouth_hf
from openvoicechat.tts.tts_melo import Mouth_melo
# from openvoicechat.tts.tts_parler import Mouth_parler
from openvoicechat.llm.llm_gpt import Chatbot_gpt
#from openvoicechat.llm.llm_llama import Chatbot_llama
# from openvoicechat.llm.llm_hf import Chatbot
from openvoicechat.stt.stt_hf import Ear_hf
from openvoicechat.utils import run_chat
from openvoicechat.llm.prompts import llama_sales
import torch
from dotenv import load_dotenv
import os

load_dotenv()


if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    print("loading models... ", device)
    load_dotenv()
    ear = Ear_hf(
        model_id="openai/whisper-tiny.en",
        silence_seconds=1.5,
        device=device,
        listen_interruptions=False,
    )

    # chatbot = Chatbot_ollama(sys_prompt=llama_sales, model="qwen2:0.5b")
    chatbot = Chatbot_gpt(sys_prompt=llama_sales, Model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
    #chatbot = Chatbot_llama(model_path="models/qwen2.5-0.5b-instruct-q8_0.gguf")

    mouth = Mouth_melo(device=device)
    #mouth = Mouth_xtts(device=device)

    run_chat(
        mouth, ear, chatbot, verbose=True, stopping_criteria=lambda x: "[END]" in x
    )
