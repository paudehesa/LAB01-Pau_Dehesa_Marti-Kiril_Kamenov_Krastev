# P0 - carga de frases en PostgreSQL
#
# Baja unas 10.000 frases, las mete en la tabla "sentences" y mide cuánto
# tarda cada INSERT. Los datos de conexión se leen del .env (DB_HOST,
# DB_PORT, DB_NAME, DB_USER y DB_PASSWORD).
#
# Ojo: el enunciado pide bookCorpus, pero con las versiones actuales de la
# librería "datasets" ya no se puede cargar (quitaron los datasets con
# script), así que usamos wikitext, que también es texto en inglés.

import os
import json
import time
import statistics as stats

import psycopg2
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()

DB_CONFIG = {
    "host": os.environ["DB_HOST"],
    "port": os.environ.get("DB_PORT", "5432"),
    "dbname": os.environ["DB_NAME"],
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
}

TARGET_SENTENCES = 10_000
MIN_SENT_LEN = 20   # con menos de 20 caracteres casi todo son títulos o ruido
MAX_SENT_LEN = 300  # y con más de 300 ya son párrafos, no frases


def get_sentences(target=TARGET_SENTENCES):
    print("Descargando frases de wikitext, esto puede tardar un poco...")
    # streaming para no bajarme el corpus entero
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)

    sentences = []
    seen = set()  # para no guardar frases repetidas
    for example in ds:
        text = example["text"].strip()
        if MIN_SENT_LEN <= len(text) <= MAX_SENT_LEN and text not in seen:
            seen.add(text)
            sentences.append(text)
            if len(sentences) >= target:
                break
    return sentences


def create_table(cur):
    # empiezo siempre de cero para que las mediciones sean comparables
    cur.execute("DROP TABLE IF EXISTS sentences;")
    cur.execute("""
        CREATE TABLE sentences (
            id          SERIAL PRIMARY KEY,
            text        TEXT NOT NULL,
            char_length INTEGER NOT NULL
        );
    """)


def load_sentences(cur, conn, sentences):
    insert_times = []
    for sent in sentences:
        t0 = time.perf_counter()
        cur.execute(
            "INSERT INTO sentences (text, char_length) VALUES (%s, %s);",
            (sent, len(sent)),
        )
        # un commit por fila a propósito: así cada tiempo incluye guardar de
        # verdad en disco y no solo mandar el INSERT
        conn.commit()
        t1 = time.perf_counter()
        insert_times.append(t1 - t0)
    return insert_times


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
    sentences = get_sentences()
    print(f"Me han salido {len(sentences)} frases válidas.")

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    create_table(cur)
    conn.commit()

    print("Insertando frases en PostgreSQL...")
    insert_times = load_sentences(cur, conn, sentences)
    summary = summarize(insert_times, "P0 - Inserción de texto")

    with open("p0_text_insert_times.json", "w") as f:
        json.dump({"summary": summary, "raw_times": insert_times}, f, indent=2)

    # guardo las frases también en un txt para que Chroma y Pgvector usen
    # exactamente las mismas
    with open("sentences.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(sentences))

    cur.close()
    conn.close()
    print(f"\nListo: la tabla 'sentences' ya tiene {len(sentences)} frases.")


if __name__ == "__main__":
    main()