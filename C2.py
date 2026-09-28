# C2 - top-2 de frases similares en Chroma
#
# Mismas 10 frases ancla que en P2, una query() por ancla y colección
# (coseno y euclidiana). La ancla siempre sale como su propio vecino más
# cercano, así que pedimos top_k+1 y la descartamos. Solo cronometramos la
# query() en sí, no el lookup del vector de la ancla.

import json
import time
import statistics as stats
from pathlib import Path

import chromadb

BASE_DIR = Path(__file__).parent
CHROMA_DB_PATH = str(BASE_DIR / "chroma_data")

# mismos ids que las 10 frases ancla elegidas en P2
ANCHOR_IDS = [1825, 410, 4507, 4013, 3658, 2287, 1680, 8936, 1425, 9675]

TOP_K = 2

COLLECTIONS = {
    "cosine": "sentences_cosine",
    "euclidean": "sentences_euclidean",
}


def query_top_k_for_collection(collection, anchor_ids, top_k):
    anchor_id_strs = [str(a) for a in anchor_ids]

    anchors_data = collection.get(ids=anchor_id_strs, include=["embeddings", "documents"])
    anchor_lookup = {
        aid: {"embedding": emb, "text": doc}
        for aid, emb, doc in zip(anchors_data["ids"], anchors_data["embeddings"], anchors_data["documents"])
    }

    results = []
    times = []

    for anchor_id_str in anchor_id_strs:
        anchor_embedding = anchor_lookup[anchor_id_str]["embedding"]
        anchor_text = anchor_lookup[anchor_id_str]["text"]

        t0 = time.perf_counter()
        query_result = collection.query(
            query_embeddings=[anchor_embedding],
            n_results=top_k + 1,  # +1 porque la propia ancla sale como su vecino más cercano
        )
        t1 = time.perf_counter()
        times.append(t1 - t0)

        neighbor_ids = query_result["ids"][0]
        neighbor_docs = query_result["documents"][0]
        neighbor_dists = query_result["distances"][0]

        neighbors = []
        for nid, ndoc, ndist in zip(neighbor_ids, neighbor_docs, neighbor_dists):
            if nid == anchor_id_str:
                continue
            neighbors.append({"id": nid, "text": ndoc, "distance": float(ndist)})
            if len(neighbors) >= top_k:
                break

        results.append(
            {"anchor_id": anchor_id_str, "anchor_text": anchor_text, "neighbors": neighbors}
        )

    return results, times


def summarize(times, label):
    summary = {
        "label": label, "n": len(times), "min": min(times), "max": max(times),
        "avg": stats.mean(times), "std": stats.stdev(times) if len(times) > 1 else 0.0,
        "total": sum(times),
    }
    print(f"\n--- {label} ---")
    print(f"operaciones:         {summary['n']}")
    print(f"mínimo:              {summary['min']:.6f} s")
    print(f"máximo:              {summary['max']:.6f} s")
    print(f"media:               {summary['avg']:.6f} s")
    print(f"desviación estándar: {summary['std']:.6f} s")
    print(f"tiempo total:        {summary['total']:.2f} s")
    return summary


def main():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    all_results = {}
    all_summaries = {}
    all_times = {}

    for metric_name, collection_name in COLLECTIONS.items():
        print(f"\nConsultando métrica '{metric_name}' (colección '{collection_name}')...")
        collection = client.get_collection(collection_name)

        results, times = query_top_k_for_collection(collection, ANCHOR_IDS, TOP_K)
        all_results[metric_name] = results
        all_times[metric_name] = times

        summary = summarize(times, f"C2 - Consultas top-{TOP_K} ({metric_name})")
        all_summaries[metric_name] = summary

    with open("c2_similarity_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open("c2_query_times.json", "w") as f:
        json.dump(
            {
                metric_name: {"summary": all_summaries[metric_name], "raw_times": all_times[metric_name]}
                for metric_name in COLLECTIONS
            },
            f,
            indent=2,
        )

    print("\nFin de ejecución.")


if __name__ == "__main__":
    main()