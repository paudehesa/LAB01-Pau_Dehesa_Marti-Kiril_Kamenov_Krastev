# G0 - carga de texto en PostgreSQL con Pgvector instalado
#
# Es prácticamente el mismo P0, pero en una tabla nueva ('sentences_pgvector')
# separada de la de P0-P2, para no mezclar los dos experimentos. Aquí
# todavía no se usa el tipo vector (eso llega en G1); de momento solo
# activamos la extensión y cargamos el texto, para poder comparar ese coste
# con el de P0 sin que pgvector influya todavía.

import os
import json
import time
import statistics as stats
from pathlib import Path

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

BASE_DIR = Path(__file__).parent
SENTENCES_FILE = BASE_DIR / "sentences.txt"
TABLE_NAME = "sentences_pgvector"


def load_sentences(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def setup_extension_and_table(cur, conn):
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise RuntimeError(
            "No se pudo activar la extensión 'vector'. Comprueba que pgvector "
            "esté instalado en el servidor antes de ejecutar este script."
        ) from e

    cur.execute(f"DROP TABLE IF EXISTS {TABLE_NAME};")
    cur.execute(
        f"""
        CREATE TABLE {TABLE_NAME} (
            id          INTEGER PRIMARY KEY,
            text        TEXT NOT NULL,
            char_length INTEGER NOT NULL
        );
        """
    )
    conn.commit()
    print(f"Extensión 'vector' activa. Tabla '{TABLE_NAME}' creada.")


def insert_sentences(cur, conn, sentences):
    # Commit por fila, igual que en P0, para medir el coste real incluyendo el fsync.
    times = []
    for i, text in enumerate(sentences):
        sentence_id = i + 1
        t0 = time.perf_counter()
        cur.execute(
            f"INSERT INTO {TABLE_NAME} (id, text, char_length) VALUES (%s, %s, %s);",
            (sentence_id, text, len(text)),
        )
        conn.commit()
        t1 = time.perf_counter()
        times.append(t1 - t0)

        if sentence_id % 1000 == 0:
            print(f"  ... {sentence_id} frases cargadas")

    return times


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
    sentences = load_sentences(SENTENCES_FILE)
    print(f"Frases leídas: {len(sentences)}")

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    setup_extension_and_table(cur, conn)

    print("Insertando frases...")
    times = insert_sentences(cur, conn, sentences)
    summary = summarize(times, "G0 - Inserción de texto (tabla preparada para pgvector)")

    with open("g0_text_insert_times.json", "w") as f:
        json.dump({"summary": summary, "raw_times": times}, f, indent=2)

    cur.close()
    conn.close()
    print("\nFin de ejecución.")


if __name__ == "__main__":
    main()