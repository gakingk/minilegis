# train_lora_rag.py
import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# ========== CONFIG ==========
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
MAX_SEQ = 2048
LOAD_4BIT = False
OUTPUT_DIR = "outputs_lora_rag"

# ========== CARREGAR MODELO ==========
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ,
    dtype = None,
    load_in_4bit = LOAD_4BIT,
)

# aplicar LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 5555,
)

# ========== DATASET ==========
DATA_FILES = {"train": "rag_sft_dataset.jsonl"}
dataset = load_dataset("json", data_files=DATA_FILES, split="train")

rag_prompt = """Você é um especialista em Direito brasileiro.
Responda com base apenas nas informações fornecidas, de forma concisa e técnica.

### Instrução:
{instruction}

### Pergunta e Contexto:
{input}

### Resposta:
{output}"""

EOS_TOKEN = tokenizer.eos_token or ""

def formatting_prompts_func(examples):
    texts = []
    for inst, inp, outp in zip(examples["instruction"], examples["input"], examples["output"]):
        txt = rag_prompt.format(
            instruction=inst.strip(),
            input=inp.strip(),
            output=outp.strip()
        ) + EOS_TOKEN
        texts.append(txt)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)
print(dataset[0])

# ========== TREINAMENTO ==========
sft_config = SFTConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    num_train_epochs=5,
    learning_rate=2e-5,
    logging_steps=10,
    save_total_limit=3,
    output_dir=OUTPUT_DIR,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    report_to="wandb",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ,
    packing=False,
    args=sft_config,
)

print("Iniciando treinamento...")
trainer_stats = trainer.train()
print("Treinamento finalizado:", trainer_stats)

# ========== SALVAR ==========
SAVE_DIR = "lora_model_rag"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Adapters salvos em {SAVE_DIR}/")
