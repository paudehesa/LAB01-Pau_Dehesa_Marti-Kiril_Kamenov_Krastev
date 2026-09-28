# G1 - embeddings con el tipo nativo vector de pgvector
#
# Igual que P1, pero la columna de embedding es del tipo vector nativo en
# vez de REAL[] genérico. Usamos el paquete pgvector para que psycopg2
# sepa convertir listas/arrays de Python al tipo vector automáticamente,
# sin tener que formatear nada a mano.

import os
import json
import time
import statistics as stats
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

DB_CONFIG = {
    "host": os.environ["DB_HOST"],
    "port": os.environ.get("DB_PORT", "5432"),
    "dbname": os.environ["DB_NAME"],
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
}

TABLE_NAME = "sentences_pgvector"
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def add_embedding_column(cur, conn):
    cur.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS embedding vector({EMBEDDING_DIM});")
    conn.commit()
    print(f"Columna 'embedding' (vector({EMBEDDING_DIM})) lista en '{TABLE_NAME}'.")


def load_sentences_from_db(cur):
    cur.execute(f"SELECT id, text FROM {TABLE_NAME} ORDER BY id;")
    return cur.fetchall()


def generate_and_store_embeddings(cur, conn, model, rows):
    gen_times = []
    store_times = []

    for sentence_id, text in rows:
        t0 = time.perf_counter()
        embedding = model.encode(text, show_progress_bar=False)
        t1 = time.perf_counter()
        gen_times.append(t1 - t0)

        t2 = time.perf_counter()
        cur.execute(
            f"UPDATE {TABLE_NAME} SET embedding = %s WHERE id = %s;",
            (embedding, sentence_id),
        )
        conn.commit()
        t3 = time.perf_counter()
        store_times.append(t3 - t2)

        if sentence_id % 1000 == 0:
            print(f"  ... {sentence_id} embeddings procesados")

    return gen_times, store_times


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
    print(f"Cargando el modelo '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)  # para que psycopg2 convierta listas de Python al tipo vector directamente
    cur = conn.cursor()

    add_embedding_column(cur, conn)

    rows = load_sentences_from_db(cur)
    print(f"Frases leídas: {len(rows)}")

    print("Generando embeddings y guardándolos...")
    gen_times, store_times = generate_and_store_embeddings(cur, conn, model, rows)

    gen_summary = summarize(gen_times, "G1 - Generación de embeddings")
    store_summary = summarize(store_times, "G1 - Guardado de embeddings (tipo vector nativo)")

    with open("g1_embedding_times.json", "w") as f:
        json.dump(
            {
                "generation": {"summary": gen_summary, "raw_times": gen_times},
                "storage": {"summary": store_summary, "raw_times": store_times},
            },
            f,
            indent=2,
        )

    cur.close()
    conn.close()
    print("\nFin de ejecución de G1.")


if __name__ == "__main__":
    main()