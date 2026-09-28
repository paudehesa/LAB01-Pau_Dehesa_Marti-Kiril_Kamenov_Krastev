# C0 - carga de frases en Chroma
#
# Lee las mismas frases que se cargaron en PostgreSQL (sentences.txt, lo
# genera P0) y las mete en una colección de Chroma con el mismo id que en
# Postgres (número de línea + 1, porque el SERIAL empezaba en 1). Mide lo
# que tarda cada inserción.
#
# Chroma no deja guardar un documento sin vector, así que aquí uso un vector
# de ceros como relleno. Solo sirve para medir el coste de guardar el texto
# (el equivalente a P0) sin mezclar todavía el cálculo del embedding. Con
# esos ceros no se puede buscar nada (con coseno la norma es 0), por eso C1
# borra esta colección y la crea otra vez con los embeddings de verdad.

import json
import time
import statistics as stats
from pathlib import Path

import chromadb

BASE_DIR = Path(__file__).parent
SENTENCES_FILE = BASE_DIR / "sentences.txt"
CHROMA_DB_PATH = str(BASE_DIR / "chroma_data")
COLLECTION_NAME = "sentences_cosine"
EMBEDDING_DIM = 384  # dimensión de all-MiniLM-L6-v2


def load_sentences(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    return sentences


def create_collection(client):
    # si quedó una colección de una ejecución anterior la borro, como el
    # DROP TABLE de P0
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # en Chroma la métrica se fija al crear la colección
    )


def load_sentences_into_chroma(collection, sentences):
    placeholder_embedding = [0.0] * EMBEDDING_DIM
    times = []

    for i, text in enumerate(sentences):
        doc_id = str(i + 1)  # mismo id que en PostgreSQL

        # una frase por llamada, a propósito, para poder medir cada
        # inserción por separado
        t0 = time.perf_counter()
        collection.add(
            ids=[doc_id],
            documents=[text],
            embeddings=[placeholder_embedding],
        )
        t1 = time.perf_counter()
        times.append(t1 - t0)

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1} frases cargadas...")

    return times


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
    sentences = load_sentences(SENTENCES_FILE)
    print(f"Leídas {len(sentences)} frases de {SENTENCES_FILE.name}.")

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = create_collection(client)
    print(f"Colección '{COLLECTION_NAME}' creada, de momento vacía.")

    print("Cargando las frases en Chroma, esto tarda unos minutos...")
    times = load_sentences_into_chroma(collection, sentences)
    summary = summarize(times, "C0 - Inserción de texto en Chroma (con placeholder)")

    with open("c0_text_insert_times.json", "w") as f:
        json.dump({"summary": summary, "raw_times": times}, f, indent=2)

    print(f"\nListo: la colección tiene {collection.count()} documentos.")


if __name__ == "__main__":
    main()