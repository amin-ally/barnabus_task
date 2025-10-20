import numpy as np
import pandas as pd
import yaml
import faiss
import onnxruntime
from transformers import AutoTokenizer
from pathlib import Path
import logging
from collections import defaultdict

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Evaluation parameters
K = 10  # Evaluate nDCG@K
NUM_QUERIES_PER_CLASS = 20  # How many random queries to test from each class
LABEL_MAP = {0: "safe", 1: "sensitive", 2: "hateful"}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def load_artifacts(config: dict):
    """Loads all necessary files for evaluation."""
    logger.info("Loading artifacts for retrieval evaluation...")

    # Load ONNX model for inference
    onnx_model_path = Path(config["serving_paths"]["models"]) / "model.onnx"
    if not onnx_model_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found at {onnx_model_path}. Please run 'make deploy'."
        )
    session = onnxruntime.InferenceSession(
        str(onnx_model_path), providers=["CPUExecutionProvider"]
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model"])

    # Load the entire dataset embeddings
    embeddings_file = Path(config["serving_paths"]["embeddings"]) / "embeddings.npz"
    if not embeddings_file.exists():
        raise FileNotFoundError(
            f"Embeddings file not found at {embeddings_file}. Please run 'make train' and 'make deploy'."
        )

    data = np.load(embeddings_file, allow_pickle=True)
    embeddings = data["embeddings"].astype("float32")
    texts = data["texts"]
    labels = data["labels"]

    logger.info(f"Loaded {len(embeddings)} documents from the corpus.")
    return session, tokenizer, embeddings, texts, labels


def create_faiss_index(embeddings: np.ndarray):
    """Creates and populates a FAISS index."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Using Inner Product similarity
    faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
    index.add(embeddings)
    logger.info(
        f"FAISS index created with {index.ntotal} vectors of dimension {dimension}."
    )
    return index


def get_query_embedding(text: str, tokenizer, session, max_length: int) -> np.ndarray:
    """Generates an embedding for a single query text using the ONNX model."""
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    ort_inputs = {
        "input_ids": encoding["input_ids"].cpu().numpy(),
        "attention_mask": encoding["attention_mask"].cpu().numpy(),
    }
    # The ONNX model was exported to output embeddings
    query_embedding = session.run(["embeddings"], ort_inputs)[0]
    faiss.normalize_L2(query_embedding)
    return query_embedding


def evaluate_retrieval():
    """Main function to run the retrieval quality evaluation focused on nDCG."""
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    session, tokenizer, corpus_embeddings, corpus_texts, corpus_labels = load_artifacts(
        config
    )
    index = create_faiss_index(corpus_embeddings)

    df = pd.DataFrame({"text": corpus_texts, "label": corpus_labels})

    # --- Sample Queries ---
    query_indices = []
    for label_id in LABEL_MAP.keys():
        class_indices = df[df["label"] == label_id].index
        if len(class_indices) > NUM_QUERIES_PER_CLASS:
            sampled = np.random.choice(
                class_indices, NUM_QUERIES_PER_CLASS, replace=False
            )
            query_indices.extend(sampled)

    logger.info(f"Selected a total of {len(query_indices)} queries for evaluation.")

    # --- Main Evaluation Loop ---
    results = defaultdict(list)
    qualitative_examples = {}

    for query_idx in query_indices:
        query_text = df.loc[query_idx, "text"]
        query_label = df.loc[query_idx, "label"]

        query_embedding = get_query_embedding(
            query_text, tokenizer, session, config["data"]["max_length"]
        )
        distances, retrieved_indices = index.search(query_embedding, K)
        retrieved_labels = df.loc[retrieved_indices[0], "label"].values

        # --- Calculate nDCG@K ---
        # 1. Relevance Score (1 if labels match, 0 otherwise)
        relevance = (retrieved_labels == query_label).astype(int)

        # 2. DCG (Discounted Cumulative Gain)
        dcg = np.sum(relevance / np.log2(np.arange(2, K + 2)))

        # 3. IDCG (Ideal Discounted Cumulative Gain)
        total_relevant_in_corpus = (df["label"] == query_label).sum()
        ideal_relevance = np.ones(min(K, total_relevant_in_corpus))
        idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))

        # 4. nDCG (Normalized DCG)
        ndcg_at_k = dcg / idcg if idcg > 0 else 0.0

        results[LABEL_MAP[query_label]].append(ndcg_at_k)

        # Store one example per class for qualitative review
        if query_label not in qualitative_examples:
            qualitative_examples[query_label] = {
                "query_text": query_text,
                "retrieved_indices": retrieved_indices[0],
                "distances": distances[0],
                "relevance": relevance,
            }

    # --- Print Report ---
    print("\n" + "=" * 80)
    print(f"| {'Retrieval Quality Report (nDCG@' + str(K) + ')'.center(76-2)} |")
    print("=" * 80)

    for label_name, class_results in results.items():
        avg_ndcg = np.mean(class_results)
        print(
            f"| Class: {label_name:<10} | nDCG@{K}: {avg_ndcg:.3f} | Num Queries: {len(class_results):<4} |"
        )

    overall_ndcg = np.mean([score for scores in results.values() for score in scores])
    print("-" * 80)
    print(
        f"| {'Overall'.ljust(10)} | nDCG@{K}: {overall_ndcg:.3f} | Total Queries: {len(query_indices):<4} |"
    )
    print("=" * 80)

    # --- Print Qualitative Examples ---
    print("\n" + "=" * 80)
    print(f"| {'Qualitative Examples: Analysis of Top K Retrieved Items'.center(76)} |")
    print("=" * 80)
    for label_id, example in sorted(qualitative_examples.items()):
        label_name = LABEL_MAP[label_id]
        print(f"\n--- Example for Class: '{label_name}' ---")
        print(f"QUERY: \"{example['query_text']}\"")
        print("-" * 50)
        for i, doc_idx in enumerate(example["retrieved_indices"]):
            retrieved_text = df.loc[doc_idx, "text"]
            retrieved_label_name = LABEL_MAP[df.loc[doc_idx, "label"]]
            similarity = example["distances"][i]
            is_hit = example["relevance"][i] == 1

            status = "✅ HIT " if is_hit else "❌ MISS"

            print(f"{i+1}. {status} (Similarity: {similarity:.4f})")
            print(
                f'   Label: {retrieved_label_name} | Text: "{retrieved_text[:100]}..."'
            )
        print("-" * 20)
    print("\n" + "=" * 80)


if __name__ == "__main__":
    evaluate_retrieval()
