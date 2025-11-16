# eval_rag_pipeline.py
import os, json, time, math
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import numpy as np

from prepare_retrieval import load_index, retrieve, EMBEDDING_MODEL
from infer_rag import gerar_resposta

try:
    from ragas import Ragas, trace_utils
    RAGAS_AVAILABLE = True
except Exception:
    RAGAS_AVAILABLE = False

try:
    from bert_score import score as bertscore
except Exception:
    bertscore = None

# CONFIG
DATASET_JSONL = "rag_sft_dataset.jsonl"
TOP_K_EVAL = 5
EMBEDDER = SentenceTransformer(EMBEDDING_MODEL)

def load_test_examples(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            pergunta = None
            inp = j.get("input","")
            if "Pergunta:" in inp:
                pergunta = inp.split("Pergunta:")[-1].split("Contexto:")[0].strip()
            else:
                pergunta = j.get("instruction","").strip()
            examples.append({
                "raw": j,
                "pergunta": pergunta,
                "referencia": j.get("output","").strip(),
                "lei": j.get("lei_nome"),
                "tipo": j.get("tipo_pergunta")
            })
    return examples

# Retrieval metrics helpers
def recall_at_k(true_doc_ids, retrieved_ids, k):
    retrieved = retrieved_ids[:k]
    return 1.0 if any(t in retrieved for t in true_doc_ids) else 0.0

def mrr_single(true_doc_ids, retrieved_ids):
    for rank, doc in enumerate(retrieved_ids, start=1):
        if doc in true_doc_ids:
            return 1.0 / rank
    return 0.0

# Main evaluation loop
def evaluate_all():
    index, metadata = load_index()
    examples = load_test_examples(DATASET_JSONL)

    stats = {
        "recall@1": [], "recall@5": [], "mrr": [],
        "gen_length": [], "latency_total": []
    }

    traces = []

    for ex in tqdm(examples, desc="Avaliar exemplos"):
        q = ex["pergunta"]
        t0 = time.time()
        retrieved = retrieve(q, k=TOP_K_EVAL, index=index, metadata=metadata, embedder=EMBEDDER)
        retrieved_ids = []
        for r in retrieved:
            m = r["meta"]
            docid = f"{m.get('lei_nome')}|{m.get('numero')}|{m.get('chunk_id','0')}"
            retrieved_ids.append(docid)

        true_doc_ids = []
        if ex.get("lei") and ex["raw"].get("artigo_numero"):
            true_doc_ids.append(f"{ex['lei']}|{ex['raw'].get('artigo_numero')}|0")

        stats["recall@1"].append(recall_at_k(true_doc_ids, retrieved_ids, 1))
        stats["recall@5"].append(recall_at_k(true_doc_ids, retrieved_ids, 5))
        stats["mrr"].append(mrr_single(true_doc_ids, retrieved_ids))

        try:
            resposta, contexto = gerar_resposta(q, k=2)
        except Exception as e:
            print("Erro geração:", e)
            resposta = ""
            contexto = ""

        t1 = time.time()
        stats["latency_total"].append(t1 - t0)
        stats["gen_length"].append(len(resposta.split()))

        bs = None
        if bertscore is not None and resposta.strip() and ex["referencia"].strip():
            P, R, F1 = bertscore([resposta], [ex["referencia"]], lang="en" if False else "pt")
            bs = float(F1.mean().cpu().numpy())

        if RAGAS_AVAILABLE:
            trace = {
                "input": q,
                "reference": ex["referencia"],
                "generated": resposta,
                "retrieved": retrieved_ids,
                "context": contexto
            }
            traces.append(trace)

    def mean(xs): return sum(xs)/len(xs) if xs else None
    summary = {
        "recall@1": mean(stats["recall@1"]),
        "recall@5": mean(stats["recall@5"]),
        "mrr": mean(stats["mrr"]),
        "avg_latency_s": mean(stats["latency_total"]),
        "avg_gen_tokens": mean(stats["gen_length"]),
    }
    print("Resumo:", json.dumps(summary, indent=2, ensure_ascii=False))

    if RAGAS_AVAILABLE and traces:
        ragas = Ragas()
        ragas.add_traces(traces)
        results = ragas.run_metrics(["context_precision","context_recall", "context_relevance"
                                     "answer_relevancy","faithfulness"])
        print("Ragas results:", results)

    with open("eval_summary.json","w",encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    evaluate_all()
