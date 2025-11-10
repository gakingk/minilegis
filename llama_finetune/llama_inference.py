from unsloth import FastLanguageModel
from transformers import TextStreamer

# ===== CONFIGURAÇÕES =====
MODEL_PATH = "lora_model"  # ou "outputs" se esse for o diretório do modelo treinado
max_seq_length = 4096      # 100000 é desnecessário para inferência normal
dtype = None               # auto detecta melhor formato
load_in_4bit = True        # reduz uso de memória (mantém performance boa)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

FastLanguageModel.for_inference(model)

alpaca_prompt = """Abaixo está uma instrução que descreve uma tarefa, acompanhada de um input que fornece o contexto necessário. 
Escreva uma resposta completa, clara e útil, explicando seu raciocínio quando apropriado.

### Instrução:
{}

### Entrada:
{}

### Resposta:
{}"""

EOS_TOKEN = tokenizer.eos_token or ""

pergunta = "O que diz o artigo 29 da Consolidação das Leis do Trabalho?"

formatted_prompt = alpaca_prompt.format(
    "Explique o seguinte trecho da legislação:",
    pergunta,
    ""
)

inputs = tokenizer([formatted_prompt], return_tensors="pt").to(model.device)

text_streamer = TextStreamer(tokenizer)

_ = model.generate(
    **inputs,
    streamer=text_streamer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
)
