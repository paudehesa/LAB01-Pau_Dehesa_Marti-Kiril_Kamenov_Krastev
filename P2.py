# P2 - top-2 de frases parecidas en PostgreSQL (sin Pgvector)
#
# Trae todos los embeddings de la tabla a Python, elige 10 frases "ancla" y
# para cada una busca las 2 frases más parecidas con dos distancias: coseno
# y euclídea. Mide lo que tarda cada búsqueda y saca mínimo, máximo, media y
# desviación estándar por métrica.
#
# Aquí Postgres no sabe calcular distancias entre vectores, así que todo el
# cálculo lo hago yo con numpy una vez me he traído los embeddings. Es justo
# donde más se nota el impedance mismatch.

import os
import json
import time
import random
import statistics as stats
from pathlib import Path

import numpy as np
import psycopg2
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

DB_CONFIG = {
    "host": os.environ["DB_HOST"],
    "port": os.environ.get("DB_PORT", "5432"),
    "dbname": os.environ["DB_NAME"],
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
}

N_ANCHORS = 10
SEED = 42  # semilla fija para que las anclas salgan siempre las mismas
TOP_K = 2


def load_all_embeddings(cur):
    print("Trayendo los embeddings desde PostgreSQL...")
    # traerlos es cosa de una sola vez y no entra en los tiempos de las
    # búsquedas, solo se imprime para tenerlo de referencia
    t0 = time.perf_counter()
    cur.execute("SELECT id, text, embedding FROM sentences ORDER BY id;")
    rows = cur.fetchall()
    t1 = time.perf_counter()
    print(f"  {len(rows)} filas traídas en {t1 - t0:.3f} s")

    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    embeddings = np.array([r[2] for r in rows], dtype=np.float32)
    return ids, texts, embeddings


def choose_anchors(ids, n_anchors, seed):
    random.seed(seed)
    # devuelve posiciones dentro de la matriz, no ids
    return random.sample(range(len(ids)), n_anchors)


def cosine_distances(matrix, vector):
    # 1 - similitud coseno, calculado para toda la matriz de golpe
    matrix_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)
    similarities = (matrix @ vector) / (matrix_norms * vector_norm + 1e-10)  # el 1e-10 evita dividir entre 0
    return 1.0 - similarities


def euclidean_distances(matrix, vector):
    diff = matrix - vector
    return np.linalg.norm(diff, axis=1)


METRICS = {
    "cosine": cosine_distances,
    "euclidean": euclidean_distances,
}


def compute_top_k_for_anchors(ids, texts, embeddings, anchor_positions, top_k):
    results = {metric_name: [] for metric_name in METRICS}
    times = {metric_name: [] for metric_name in METRICS}

    for metric_name, distance_fn in METRICS.items():
        for pos in anchor_positions:
            anchor_id = ids[pos]
            anchor_text = texts[pos]
            anchor_vector = embeddings[pos]

            # solo cronometro el cálculo de distancias y el top-k
            t0 = time.perf_counter()
            dists = distance_fn(embeddings, anchor_vector)
            dists[pos] = np.inf  # la propia frase a infinito, para que no salga como su vecina
            nearest_positions = np.argsort(dists)[:top_k]
            t1 = time.perf_counter()

            times[metric_name].append(t1 - t0)

            neighbors = [
                {"id": int(ids[p]), "text": texts[p], "distance": float(dists[p])}
                for p in nearest_positions
            ]
            results[metric_name].append(
                {"anchor_id": int(anchor_id), "anchor_text": anchor_text, "neighbors": neighbors}
            )

    return results, times


def summarize(times, label):
    summary = {
        "label": label,
        "n": len(times),
        "min": min(times),
        "max": max(times),
        "avg": stats.mean(times),
        "std": stats.stdev(times) if len(times) > 1 else 0.0,
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
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    ids, texts, embeddings = load_all_embeddings(cur)
    print(f"Matriz de embeddings: {embeddings.shape}")

    anchor_positions = choose_anchors(ids, N_ANCHORS, SEED)
    anchor_ids = [ids[p] for p in anchor_positions]
    print(f"\nFrases ancla elegidas (ids): {anchor_ids}")
    for p in anchor_positions:
        print(f"  id={ids[p]}: \"{texts[p][:80]}...\"")

    print("\nBuscando las 2 frases más parecidas a cada ancla...")
    results, times = compute_top_k_for_anchors(ids, texts, embeddings, anchor_positions, TOP_K)

    all_summaries = {}
    for metric_name in METRICS:
        summary = summarize(times[metric_name], f"P2 - Consultas top-{TOP_K} ({metric_name})")
        all_summaries[metric_name] = summary

    with open("p2_similarity_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open("p2_query_times.json", "w") as f:
        json.dump(
            {
                metric_name: {"summary": all_summaries[metric_name], "raw_times": times[metric_name]}
                for metric_name in METRICS
            },
            f,
            indent=2,
        )

    cur.close()
    conn.close()
    print("\nListo. Vecinos en p2_similarity_results.json y tiempos en p2_query_times.json.")


if __name__ == "__main__":
    main()