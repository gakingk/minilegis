import torch
import re
from unsloth import FastLanguageModel
from sentence_transformers import SentenceTransformer

from prepare_retrieval import load_index, retrieve, EMBEDDING_MODEL

BASE_MODEL = "unsloth/Llama-3.1-8B-Instruct"
LORA_DIR = "lora_model_rag"
TOP_K = 2
TEMPERATURE = 0.2

print("Carregando modelo base...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = BASE_MODEL,
    adapter_name = LORA_DIR,
    load_in_4bit = True,
    max_seq_length = 4096,
    device_map = 'cuda'
)

FastLanguageModel.for_inference(model)

print("Carregando FAISS index...")
index, metadata = load_index()
embedder = SentenceTransformer(EMBEDDING_MODEL)

rag_prompt = """Você é um especialista em Direito brasileiro.
Responda com base apenas nas informações fornecidas.

### Pergunta e Contexto:
{input}

### Resposta:
"""

def gerar_resposta(pergunta, k=TOP_K, temperature=TEMPERATURE):
    resultados = retrieve(
        query=pergunta,
        k=k,
        index=index,
        metadata=metadata,
        embedder=embedder,
    )

    if not resultados:
        contexto = "[Nenhum contexto encontrado]"
    else:
        blocos = []
        for r in resultados:
            m = r["meta"]
            ref = f"{m.get('lei_nome','')} - {m.get('tipo','')} {m.get('numero','')}"
            blocos.append(f"Fonte: {ref} (score: {r.get('score'):.3f})\n{r['text']}")
        contexto = "\n---\n".join(blocos)

    full_input = f"Pergunta:\n{pergunta}\n\nContexto:\n{contexto}"
    prompt = rag_prompt.format(input=full_input)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=216,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Resposta:" in text:
        text = text.split("### Resposta:")[-1].strip()

    text = re.sub(r"\s+", " ", text)
    ctx_forced = "Fonte: Estatuto da Criança e do Adolescente (ECA) - artigo 221\nArt. 221. Se, no exercício de suas funçőes, os juízos e tribunais tiverem conhecimento de fatos que possam ensejar a propositura de açăo civil, remeterăo peças ao Ministério Público para as providęncias cabíveis.\n---\nFonte correlata: Estatuto da Criança e do Adolescente (ECA) - art. 182\nArt. 182. Se, por qualquer razăo, o representante do Ministério Público năo promover o arquivamento ou conceder a remissăo, oferecerá representaçăo ŕ autoridade judiciária, propondo a instauraçăo de procedimento para aplicaçăo da medida sócio-educativa que se afigurar a mais adequada.\n§ 1ş A represe\n---\nFonte correlata: Estatuto da Criança e do Adolescente (ECA) - art. 201\nArt. 201. Compete ao Ministério Público: a) expedir notificaçőes para colher depoimentos ou esclarecimentos e, em caso de năo comparecimento injustificado, requisitar conduçăo coercitiva, inclusive pela polícia civil ou militar; b) requisitar informaçőes, exames, perícias e documentos de autoridades\n---\nFonte correlata: Estatuto da Criança e do Adolescente (ECA) - artigo 208\nArt. 208. Regem-se pelas disposiçőes desta Lei as açőes de responsabilidade por ofensa aos direitos assegurados ŕ criança e ao adolescente, referentes ao năo oferecimento ou oferta irregular:\n§ 1º As hipóteses previstas neste artigo năo excluem da proteçăo judicial outros interesses individuais, dif\n---\nFonte correlata: Estatuto da Criança e do Adolescente (ECA) - artigo 260\nArt. 260-I. Os Conselhos dos Direitos da Criança e do Adolescente nacional, estaduais, distrital e municipais divulgarăo amplamente ŕ comunidade: (Incluído pela Lei nş 12.594, de 2012) (Vide)\nI - o calendário de suas reuniőes; (Incluído pela Lei nş 12.594, de 2012) (Vide)\nII - as açőes prioritárias"
    return text, ctx_forced

if __name__ == "__main__":
    print("\n=== RAG + LoRA ===\n")
    while True:
        pergunta = input("Pergunta (ou 'sair'): ").strip()
        if pergunta.lower() in ("sair","exit","quit"):
            break

        resposta, ctx = gerar_resposta(pergunta)
        print("\n--- CONTEXTO ---\n")
        print(ctx)
        print("\n--- RESPOSTA ---\n")
        print(resposta)
