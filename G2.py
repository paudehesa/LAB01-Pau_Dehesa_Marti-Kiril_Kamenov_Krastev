# G2 - top-2 de frases similares con SQL nativo de pgvector
#
# Mismas 10 frases ancla que en P2/C2. A diferencia de P2, aquí no hace
# falta traer los embeddings a Python y calcular distancias a mano: se usa
# directamente el operador nativo de pgvector en la consulta SQL (<=> para
# coseno, <-> para euclidiana). No se crea ningún índice (HNSW/IVFFlat),
# así que la búsqueda es exacta por fuerza bruta, igual que en P2.

import os
import json
import time
import statistics as stats
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

DB_CONFIG = {
    "host": os.environ["DB_HOST"],
    "port": os.environ.get("DB_PORT", "5432"),
    "dbname": os.environ["DB_NAME"],
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
}

TABLE_NAME = "sentences_pgvector"

# mismos ids que en P2/C2
ANCHOR_IDS = [1825, 410, 4507, 4013, 3658, 2287, 1680, 8936, 1425, 9675]
TOP_K = 2

METRIC_OPERATORS = {
    "cosine": "<=>",
    "euclidean": "<->",
}


def load_anchor_vectors(cur, anchor_ids):
    cur.execute(
        f"SELECT id, text, embedding FROM {TABLE_NAME} WHERE id = ANY(%s);",
        (anchor_ids,),
    )
    rows = cur.fetchall()
    return {row[0]: {"text": row[1], "embedding": row[2]} for row in rows}


def query_top_k(cur, anchor_lookup, anchor_ids, operator, top_k):
    results = []
    times = []

    for anchor_id in anchor_ids:
        anchor_text = anchor_lookup[anchor_id]["text"]
        anchor_vector = anchor_lookup[anchor_id]["embedding"]

        query = f"""
            SELECT id, text, (embedding {operator} %s) AS distance
            FROM {TABLE_NAME}
            WHERE id != %s
            ORDER BY distance
            LIMIT %s;
        """

        t0 = time.perf_counter()
        cur.execute(query, (anchor_vector, anchor_id, top_k))
        neighbors_rows = cur.fetchall()
        t1 = time.perf_counter()
        times.append(t1 - t0)

        neighbors = [
            {"id": nid, "text": ntext, "distance": float(ndist)}
            for nid, ntext, ndist in neighbors_rows
        ]
        results.append({"anchor_id": anchor_id, "anchor_text": anchor_text, "neighbors": neighbors})

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
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    cur = conn.cursor()

    anchor_lookup = load_anchor_vectors(cur, ANCHOR_IDS)
    print(f"Frases ancla cargadas: {list(anchor_lookup.keys())}")

    all_results = {}
    all_summaries = {}
    all_times = {}

    for metric_name, operator in METRIC_OPERATORS.items():
        print(f"\nConsultando métrica '{metric_name}' (operador '{operator}')...")
        results, times = query_top_k(cur, anchor_lookup, ANCHOR_IDS, operator, TOP_K)
        all_results[metric_name] = results
        all_times[metric_name] = times
        all_summaries[metric_name] = summarize(times, f"G2 - Consultas top-{TOP_K} ({metric_name})")

    with open("g2_similarity_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open("g2_query_times.json", "w") as f:
        json.dump(
            {
                metric_name: {"summary": all_summaries[metric_name], "raw_times": all_times[metric_name]}
                for metric_name in METRIC_OPERATORS
            },
            f,
            indent=2,
        )

    cur.close()
    conn.close()
    print("\nFin de ejecución.")


if __name__ == "__main__":
    main()