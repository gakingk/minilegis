import os, json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# CONFIG
PASTA_JSON = "../leis_json"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
INDEX_PATH = "faiss_index.idx"
METADATA_PATH = "faiss_metadata.json"
DOCS_PATH = "documents.json"  # lista de chunks com metadados
CHUNK_SIZE_CHARS = 1600
CHUNK_OVERLAP = 200
BATCH = 128


def chunk_text(text, chunk_size=CHUNK_SIZE_CHARS, overlap=CHUNK_OVERLAP):
    """Divide texto em blocos sobrepostos."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def combine_with_children(item):
    """
    Combina o texto principal do artigo com seus parágrafos e incisos.
    """
    partes = [item.get("texto", "").strip()]

    # Adiciona parágrafos
    for p in item.get("paragrafos", []):
        partes.append(p.get("texto", "").strip())
        for inc in p.get("incisos", []):
            partes.append(inc.get("texto", "").strip())

    # Adiciona incisos diretos
    for inc in item.get("incisos", []):
        partes.append(inc.get("texto", "").strip())

    return "\n".join([p for p in partes if p])


def load_articles():
    docs = []
    for fname in os.listdir(PASTA_JSON):
        if not fname.endswith(".json"):
            continue

        with open(os.path.join(PASTA_JSON, fname), "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Erro lendo {fname}: {e}")
                continue

            for item in data:
                texto_completo = combine_with_children(item)
                if not texto_completo:
                    continue

                meta = {
                    "lei_nome": item.get("lei_nome"),
                    "lei_numero": item.get("lei_numero"),
                    "tipo": item.get("tipo"),
                    "numero": item.get("numero"),
                    "url": item.get("url"),
                    "source_file": fname
                }

                chunks = chunk_text(texto_completo)
                for i, ch in enumerate(chunks):
                    docs.append({"meta": {**meta, "chunk_id": i}, "text": ch})

    print(f"Total de chunks coletados: {len(docs)}")
    return docs


def build_index(docs):
    print("Carregando modelo de embeddings:", EMBEDDING_MODEL)
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    d = embedder.get_sentence_embedding_dimension()

    index = faiss.IndexFlatIP(d)
    metadata = []

    texts = [dct["text"] for dct in docs]
    for i in tqdm(range(0, len(texts), BATCH), desc="Embed batches"):
        batch = texts[i:i+BATCH]
        embs = embedder.encode(batch, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        index.add(embs)
        metadata.extend([docs[j] for j in range(i, i+len(batch))])

    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print("Index salvo em:", INDEX_PATH)
    print("Metadados salvos em:", METADATA_PATH)
    return index, metadata


def load_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("Index ou metadados não encontrados.")
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def retrieve(query, k=5, index=None, metadata=None, embedder=None):
    """
    Recupera os k trechos mais relevantes
    """
    if index is None or metadata is None or embedder is None:
        index, metadata = load_index()
        embedder = SentenceTransformer(EMBEDDING_MODEL)

    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb, k * 4)  # busca mais e filtra

    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata):
            continue

        m = metadata[idx]["meta"]
        txt = metadata[idx]["text"]

        # Evita colocar artigo primeiro em tudo
        if str(m.get("numero")).strip() in ("1", "I", "Art. 1º"):
            continue

        # Filtro de similaridade mínimo
        if score < 0.25:
            continue

        results.append({"score": float(score), "meta": m, "text": txt})

    # Remove duplicados por lei/artigo
    unique = {}
    for r in results:
        key = (r["meta"]["lei_nome"], r["meta"]["numero"])
        if key not in unique:
            unique[key] = r

    results = sorted(unique.values(), key=lambda x: -x["score"])[:k]
    return results


if __name__ == "__main__":
    docs = load_articles()
    build_index(docs)

