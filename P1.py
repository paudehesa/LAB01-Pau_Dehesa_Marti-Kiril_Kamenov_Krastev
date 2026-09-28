# P1 - embeddings en PostgreSQL
#
# Coge las frases que dejó P0 en la tabla "sentences", genera el embedding
# de cada una con all-MiniLM-L6-v2 y lo guarda en una columna nueva. Como
# aquí no se puede usar Pgvector, el vector va en una columna REAL[] (un
# array de floats normal de Postgres).
# Mide por separado lo que tarda generar el embedding y lo que tarda
# guardarlo. Lee la conexión del .env, igual que P0.

import os
import json
import time
import statistics as stats
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

DB_CONFIG = {
    "host": os.environ["DB_HOST"],
    "port": os.environ.get("DB_PORT", "5432"),
    "dbname": os.environ["DB_NAME"],
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
}

MODEL_NAME = "all-MiniLM-L6-v2"


def add_embedding_column(cur, conn):
    cur.execute("ALTER TABLE sentences ADD COLUMN IF NOT EXISTS embedding REAL[];")
    conn.commit()
    print("Columna 'embedding' lista en la tabla 'sentences'.")


def load_sentences_from_db(cur):
    cur.execute("SELECT id, text FROM sentences ORDER BY id;")
    return cur.fetchall()


def generate_and_store_embeddings(cur, conn, model, rows):
    gen_times = []
    store_times = []

    for sentence_id, text in rows:
        # el embedding se genera frase a frase y no en lote a propósito: en
        # lote iría mucho más rápido, pero así puedo medir cada frase por
        # separado, que es lo que pide el enunciado
        t0 = time.perf_counter()
        embedding = model.encode(text, show_progress_bar=False)
        t1 = time.perf_counter()
        gen_times.append(t1 - t0)

        # UPDATE y no INSERT porque las filas ya existen desde P0
        t2 = time.perf_counter()
        cur.execute(
            "UPDATE sentences SET embedding = %s WHERE id = %s;",
            (embedding.tolist(), sentence_id),
        )
        conn.commit()  # commit por fila, igual que en P0, para que los tiempos sean comparables
        t3 = time.perf_counter()
        store_times.append(t3 - t2)

        if sentence_id % 1000 == 0:
            print(f"  {sentence_id} frases procesadas...")

    return gen_times, store_times


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
    print(f"Cargando el modelo {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    add_embedding_column(cur, conn)

    rows = load_sentences_from_db(cur)
    print(f"Hay {len(rows)} frases en la tabla.")

    print("Generando y guardando los embeddings, esto tarda un par de minutos...")
    gen_times, store_times = generate_and_store_embeddings(cur, conn, model, rows)

    gen_summary = summarize(gen_times, "P1 - Generación de embeddings")
    store_summary = summarize(store_times, "P1 - Guardado de embeddings en PostgreSQL")

    with open("p1_embedding_times.json", "w") as f:
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
    print("\nListo: todas las frases de 'sentences' ya tienen su embedding.")


if __name__ == "__main__":
    main()