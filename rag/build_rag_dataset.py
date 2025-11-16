import os, json, random, torch, re
import numpy as np
from tqdm import tqdm
from unsloth import FastLanguageModel
from prepare_retrieval import load_index, retrieve, SentenceTransformer, EMBEDDING_MODEL

# ======================
# CONFIG
# ======================
PASTA_JSON = "../leis_json"
OUTPUT_JSONL = "rag_sft_dataset.jsonl"
QUESTIONS_PER_ARTICLE = 3
TOP_K = 2
TOP_K_CORRELATOS = 2
USE_INTERPRETACOES = True
MAX_CONTEXT_CHARS = 4500
TEMPERATURE = 0.2
LOAD_4BIT = True
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
MAX_SEQ_LEN = 2048

# ======================
# MODELO LLAMA LOCAL
# ======================
print("Carregando modelo Unsloth para geração local")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LEN,
    dtype = None,
    load_in_4bit = LOAD_4BIT,
)
FastLanguageModel.for_inference(model)

# ======================
# FUNÇÃO DE INFERÊNCIA LOCAL
# ======================
def gerar_interpretacao(pergunta, contexto, lei, numero, temperature=TEMPERATURE):
    """Gera uma resposta interpretativa localmente com controle anti-alucinação."""

    if len(contexto) > 1500:
        contexto = contexto[:1500] + " ... [trecho truncado]"
    
    prompt = f"""Você é um especialista em Direito brasileiro.
Responda à pergunta somente com base no contexto abaixo.
Não copie o contexto nem repita a pergunta.
Responda de forma direta e concisa. Use no máximo um parágrafo.
Não mencione artigos, incisos, leis ou dispositivos que não aparecem no contexto.
Por exemplo, se o contexto não mencionar a Constituição Federal, não a cite.
Mantenha um tom jurídico, técnico e interpretativo, mas limitado ao conteúdo visível.

### Pergunta:
{pergunta}

### Contexto:
\"\"\"{contexto}\"\"\"

### Resposta:"""

    # Tokenização e geração
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decodifica e limpa
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    text = re.split(r"###\s*(Pergunta|Contexto|Resposta)\s*:", text)[-1].strip()

    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(O artigo\s+\d+[^.]*\.)\s*(\1)+", r"\1", text)  # remove repetições
    text = re.sub(r"(\b[oO] artigo\s+\d+.*?)(?=\b[oO] artigo\s+\d+)", r"\1.", text)
    lower_text = text.lower()

    # --- Filtros anti-alucinação ---
    # Menciona artigo diferente
    # if re.search(r"art\.?\s*\d+", lower_text):
    #     numeros_citados = re.findall(r"art\.?\s*(\d+)", lower_text)
    #     if any(n != str(numero) for n in numeros_citados):
    #         print(f"[Aviso] Possível alucinação (artigo incorreto) em {lei} - art. {numero}")
    #         return None
    # Estava removendo artigos desejáveis para resposta
        
    # --- Pós-processamento ---
    sentencas = re.split(r"(?<=[.!?])\s+", text)
    if len(sentencas) > 4:  # corta em 1 parágrafo real
        text = " ".join(sentencas[:4])
        if not text.endswith("."):
            text += "."

    # remove repetições
    frases = []
    for s in sentencas:
        if s not in frases:
            frases.append(s)
    text = " ".join(frases[:4]).strip()

    # Remove repetições quase idênticas de parágrafos (modelo ecoando o início)
    text = re.sub(r"([^.]{30,}\.)\s+\1+", r"\1", text)

    # Se o modelo repetiu o início da resposta
    if len(text) > 100:
        first_part = text[:120]
        rest = text[120:]
        if first_part in rest:
            text = first_part + rest.split(first_part)[-1]


    # Resposta curta demais (modelo travou ou retornou prompt)
    if len(text.split()) < 20:
        print(f"[Aviso] Resposta curta demais em {lei} - art. {numero}")
        return None

    return text
# ======================
# AUXILIARES
# ======================
def iter_articles():
    for fname in os.listdir(PASTA_JSON):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(PASTA_JSON, fname), "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"[Erro ao ler {fname}]:", e)
                continue
            for item in data:
                if item.get("tipo") == "artigo" and item.get("texto"):
                    yield item

def montar_texto_completo(art):
    partes = [art.get("texto", "").strip()]
    for p in art.get("paragrafos", []):
        partes.append(p.get("texto", "").strip())
        for inc in p.get("incisos", []):
            partes.append(inc.get("texto", "").strip())
    for inc in art.get("incisos", []):
        partes.append(inc.get("texto", "").strip())
    texto = "\n".join([p for p in partes if p])
    return texto[:800]  # evita artigos enormes

# ======================
# CACHE DE EMBEDDINGS
# ======================
def precompute_embeddings_por_lei(artigos, embedder):
    cache = {}
    for lei_nome in set(a.get("lei_nome") for a in artigos):
        subset = [a for a in artigos if a.get("lei_nome") == lei_nome]
        textos = [montar_texto_completo(a) for a in subset]
        embs = embedder.encode(textos, convert_to_numpy=True, normalize_embeddings=True)
        cache[lei_nome] = list(zip(subset, embs))
    return cache

def artigos_correlatos_semanticos(artigo, cache_embeds, top_k=TOP_K_CORRELATOS):
    lei_nome = artigo.get("lei_nome")
    texto = montar_texto_completo(artigo)
    base_emb = embedder.encode([texto], convert_to_numpy=True, normalize_embeddings=True)[0]
    candidatos = cache_embeds[lei_nome]
    sims = [(np.dot(base_emb, emb), art) for art, emb in candidatos if art is not artigo]
    sims.sort(key=lambda x: -x[0])
    return [a for _, a in sims[:top_k]]

# ======================
# BUILD DATASET
# ======================
index, metadata = load_index()
embedder = SentenceTransformer(EMBEDDING_MODEL)
todos_artigos = list(iter_articles())
cache_embeds = precompute_embeddings_por_lei(todos_artigos, embedder)

with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
    for art in tqdm(todos_artigos, desc="Artigos"):
        numero = art.get("numero")
        lei = art.get("lei_nome")
        texto_art = montar_texto_completo(art)

        templates_sel = random.sample([
            ("factual", "O que diz o artigo {numero} da {lei}?"),
            ("interpretativo", "Explique o conteúdo do artigo {numero} da {lei}."),
            ("interpretativo", "Quais direitos ou deveres o artigo {numero} da {lei} estabelece?")
        ], k=QUESTIONS_PER_ARTICLE)

        for tipo, t in templates_sel:
            pergunta = t.format(numero=numero, lei=lei)
            alvo_ref = f"{lei} - artigo {numero}"
            alvo_entry = f"Fonte: {alvo_ref}\n{texto_art}"

            correlatos = artigos_correlatos_semanticos(art, cache_embeds)
            correlato_entries = [
                f"Fonte correlata: {c.get('lei_nome')} - art. {c.get('numero')}\n"
                + montar_texto_completo(c)[:300]
                for c in correlatos
            ]

            retrieved = retrieve(pergunta, k=TOP_K * 3, index=index, metadata=metadata, embedder=embedder)

            retrieved_entries = []
            for r in retrieved:
                meta = r["meta"]

                # só mantém artigos da mesma lei
                if meta.get("lei_nome") != lei:
                    continue

                # Evita repetir o mesmo artigo
                if str(meta.get("numero")) == str(numero):
                    continue

                ref = f"{meta.get('lei_nome', '')} - {meta.get('tipo','')} {meta.get('numero', '')}"
                retrieved_entries.append(f"Fonte correlata: {ref}\n{r['text'][:300]}")

            # Mantém só até TOP_K entradas correlatas da mesma lei
            retrieved_entries = retrieved_entries[:TOP_K]


            # ---- Compacta o contexto ----
            context_entries = [alvo_entry] + correlato_entries + retrieved_entries
            context_block = "\n---\n".join(context_entries)
            context_block = context_block[:MAX_CONTEXT_CHARS]

            # ---- Saída ----
            if tipo == "interpretativo" and USE_INTERPRETACOES:
                try:
                    output = gerar_interpretacao(pergunta, context_block, numero=numero, lei=lei)
                except Exception as e:
                    print(f"[Falha geração {lei} art. {numero}]:", e)
                    output = texto_art
            else:
                output = texto_art

            fout.write(json.dumps({
                "instruction": "Responda a pergunta utilizando apenas as informações dadas no contexto.",
                "input": f"Pergunta:\n{pergunta}\n\nContexto:\n{context_block}",
                "output": output,
                "lei_nome": lei,
                "artigo_numero": numero,
                "tipo_pergunta": tipo
            }, ensure_ascii=False) + "\n")

print(f"Dataset final salvo em: {OUTPUT_JSONL}")
