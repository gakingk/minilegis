import json
import os
from datasets import Dataset
from transformers import AutoTokenizer

# ===== CONFIGURAÇÕES =====
PASTA_JSON = "../leis_json"  # Pasta com arquivos .json individuais
DATASET_JSON = "dataset_legislativo_alpaca.json"  # Saída final (formato JSONL)

# Nome do modelo usado para tokenização — deve ser o mesmo do fine-tuning
MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"

todos_pares = []

for arquivo in os.listdir(PASTA_JSON):
    if arquivo.endswith(".json"):
        caminho = os.path.join(PASTA_JSON, arquivo)
        with open(caminho, "r", encoding="utf-8") as f:
            try:
                unidades = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Erro ao ler {arquivo}: {e}")
                continue

            for unidade in unidades:
                texto = unidade.get("texto", "").strip()
                lei_nome = unidade.get("lei_nome", "Lei")
                tipo = unidade.get("tipo", "")
                numero = unidade.get("numero", "")

                if not texto or not numero:
                    continue  # pular itens vazios

                pergunta = f"O que diz o {tipo} {numero} da {lei_nome}?"

                todos_pares.append({
                    "instruction": "Explique o seguinte trecho da legislação:",
                    "input": pergunta,
                    "output": texto
                })

print(f"Total de exemplos carregados: {len(todos_pares)}")

# ===== 2. Converter para formato HuggingFace Dataset =====
dataset = Dataset.from_list(todos_pares)

# ===== 3. Criar prompt estilo Alpaca em português =====
alpaca_prompt = """Abaixo está uma instrução que descreve uma tarefa, acompanhada de um input que fornece o contexto necessário. 
Escreva uma resposta completa, clara e útil, explicando seu raciocínio quando apropriado.

### Instrução:
{}

### Entrada:
{}

### Resposta:
{}"""

# ===== 4. Tokenizador e EOS_TOKEN =====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
EOS_TOKEN = tokenizer.eos_token or ""  # fallback se o modelo não tiver definido

def formatting_prompts_func(examples):
    texts = []
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        formatted = alpaca_prompt.format(instruction.strip(), input_text.strip(), output.strip()) + EOS_TOKEN
        texts.append(formatted)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

dataset.to_json(DATASET_JSON, orient="records", force_ascii=False)
print(f"Dataset salvo em {DATASET_JSON} com {len(dataset)} exemplos.")
