# C1 - embeddings reales en Chroma
#
# Sustituimos los vectores placeholder de C0 por los embeddings reales
# (mismo modelo que en P1) y creamos tambien la coleccion euclidiana, ya
# que en Chroma la metrica va ligada a la coleccion y no se puede cambiar.
#
# El sync_threshold alto es para esquivar un bug de Chroma en Windows que
# corrompe el indice HNSW al persistirlo en disco; asi se queda siempre en
# modo fuerza bruta, que para 10.000 vectores va sobrado de rapido.

import json
import time
import statistics as stats
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent
SENTENCES_FILE = BASE_DIR / "sentences.txt"
CHROMA_DB_PATH = str(BASE_DIR / "chroma_data")

COSINE_COLLECTION_NAME = "sentences_cosine"        # ya creada en C0
EUCLIDEAN_COLLECTION_NAME = "sentences_euclidean"  # se crea aqui

MODEL_NAME = "all-MiniLM-L6-v2"


def load_sentences(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def generate_and_update_cosine(model, client, sentences):
    # El embedding se genera una vez por frase y se reutiliza en la
    # coleccion euclidiana, asi el coste de generacion no se cuenta doble.
    try:
        client.delete_collection(COSINE_COLLECTION_NAME)
    except Exception:
        pass
    cosine_collection = client.create_collection(
        name=COSINE_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "hnsw:sync_threshold": 1_000_000},
    )

    gen_times = []
    store_cosine_times = []
    embeddings = []

    for i, text in enumerate(sentences):
        doc_id = str(i + 1)

        t0 = time.perf_counter()
        embedding = model.encode(text, show_progress_bar=False)
        t1 = time.perf_counter()
        gen_times.append(t1 - t0)
        embeddings.append(embedding.tolist())

        t2 = time.perf_counter()
        cosine_collection.add(ids=[doc_id], documents=[text], embeddings=[embedding.tolist()])
        t3 = time.perf_counter()
        store_cosine_times.append(t3 - t2)

        if (i + 1) % 1000 == 0:
            print(f"  ... {i + 1} frases procesadas (coseno)")

    return gen_times, store_cosine_times, embeddings, cosine_collection


def create_and_fill_euclidean(client, sentences, embeddings):
    try:
        client.delete_collection(EUCLIDEAN_COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=EUCLIDEAN_COLLECTION_NAME,
        metadata={"hnsw:space": "l2", "hnsw:sync_threshold": 1_000_000},
    )

    store_euclidean_times = []
    for i, (text, embedding) in enumerate(zip(sentences, embeddings)):
        doc_id = str(i + 1)

        t0 = time.perf_counter()
        collection.add(ids=[doc_id], documents=[text], embeddings=[embedding])
        t1 = time.perf_counter()
        store_euclidean_times.append(t1 - t0)

        if (i + 1) % 1000 == 0:
            print(f"  ... {i + 1} frases procesadas (euclidiana)")

    return collection, store_euclidean_times


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

    print(f"Cargando el modelo '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    print("Generando embeddings y guardando en la colección coseno...")
    gen_times, store_cosine_times, embeddings, cosine_collection = generate_and_update_cosine(
        model, client, sentences
    )

    print("Guardando los mismos embeddings en la colección euclidiana...")
    euclidean_collection, store_euclidean_times = create_and_fill_euclidean(
        client, sentences, embeddings
    )

    gen_summary = summarize(gen_times, "C1 - Generación de embeddings")
    store_cosine_summary = summarize(store_cosine_times, "C1 - Guardado en colección coseno")
    store_euclidean_summary = summarize(store_euclidean_times, "C1 - Guardado en colección euclidiana")

    with open("c1_embedding_times.json", "w") as f:
        json.dump(
            {
                "generation": {"summary": gen_summary, "raw_times": gen_times},
                "storage_cosine": {"summary": store_cosine_summary, "raw_times": store_cosine_times},
                "storage_euclidean": {"summary": store_euclidean_summary, "raw_times": store_euclidean_times},
            },
            f,
            indent=2,
        )

    print(f"\nListo. Colección '{COSINE_COLLECTION_NAME}': {cosine_collection.count()} documentos.")
    print(f"Colección '{EUCLIDEAN_COLLECTION_NAME}': {euclidean_collection.count()} documentos.")

    # Chroma va escribiendo en segundo plano, así que le damos unos segundos
    # antes de cerrar para que le dé tiempo a terminar de volcar todo a disco.
    print("\nEsperando unos segundos a que Chroma acabe de escribir...")
    time.sleep(15)
    print("Hecho.")


if __name__ == "__main__":
    main()