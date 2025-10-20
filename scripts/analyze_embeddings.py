# scripts/analyze_embeddings.py
import numpy as np
import pandas as pd
from pathlib import Path
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def plot_embeddings(
    df: pd.DataFrame, hue_column: str, palette: dict, title: str, save_path: Path
):
    """Helper function to create and save a scatter plot."""
    logging.info(f"Generating plot for '{hue_column}'...")
    plt.figure(figsize=(14, 10))

    sns.scatterplot(
        data=df,
        x="umap1",
        y="umap2",
        hue=hue_column,
        palette=palette,
        s=10,
        alpha=0.7,
        hue_order=list(palette.keys()),  # Ensure consistent legend order
    )

    plt.title(title, fontsize=18, fontweight="bold")
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.legend(title=hue_column.capitalize(), markerscale=2, fontsize=12)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logging.info(f"Plot saved to: {save_path.absolute()}")
    plt.close()


def main(embeddings_path: Path, output_dir: Path, sample_size: int):
    """Main function to load, process, and visualize embeddings."""

    # --- 1. Load Data ---
    if not embeddings_path.exists():
        logging.error(f"Embeddings file not found at {embeddings_path.absolute()}")
        logging.error("Please run `make train` first to generate embeddings.")
        return

    logging.info(f"Loading embeddings from {embeddings_path}...")
    data = np.load(embeddings_path, allow_pickle=True)
    embeddings = data["embeddings"]
    languages = data["languages"]
    labels = data["labels"]

    label_map = {0: "safe", 1: "sensitive", 2: "hateful"}
    text_labels = np.array([label_map.get(lbl, "unknown") for lbl in labels])

    logging.info(
        f"Loaded {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}."
    )

    # --- 2. Subsample Data ---
    if len(embeddings) > sample_size:
        logging.info(f"Subsampling to {sample_size} points for visualization.")
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings_sample = embeddings[indices]
        languages_sample = languages[indices]
        labels_sample = text_labels[indices]
    else:
        embeddings_sample = embeddings
        languages_sample = languages
        labels_sample = text_labels

    # --- 3. Run UMAP ---
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2, metric="cosine", random_state=42
    )

    logging.info("Running UMAP transformation... This may take a minute.")
    embeddings_2d = reducer.fit_transform(embeddings_sample)
    logging.info("UMAP transformation complete.")

    df_viz = pd.DataFrame(
        {
            "umap1": embeddings_2d[:, 0],
            "umap2": embeddings_2d[:, 1],
            "language": languages_sample,
            "label": labels_sample,
        }
    )

    # --- 4. Generate and Save Plots ---
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot by Language
    plot_embeddings(
        df=df_viz,
        hue_column="language",
        palette={"en": "cornflowerblue", "fa": "darkorange"},
        title="2D UMAP Projection of Embeddings by Language",
        save_path=output_dir / "embedding_analysis_by_language.png",
    )

    # Plot by Label
    plot_embeddings(
        df=df_viz,
        hue_column="label",
        palette={"safe": "mediumseagreen", "sensitive": "gold", "hateful": "crimson"},
        title="2D UMAP Projection of Embeddings by Class Label",
        save_path=output_dir / "embedding_analysis_by_label.png",
    )

    logging.info("Analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze and visualize multilingual sentence embeddings."
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=Path("data/embeddings/embeddings.npz"),
        help="Path to the embeddings.npz file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/plots"),
        help="Directory to save the output plots.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Number of data points to use for the visualization.",
    )
    args = parser.parse_args()

    main(args.embeddings_path, args.output_dir, args.sample_size)
