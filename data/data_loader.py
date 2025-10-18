import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultilingualHateSpeechDataLoader:
    """Load and prepare multilingual hate speech datasets (English + Farsi)"""

    def __init__(
        self,
        test_size=0.15,
        val_size=0.15,
        random_state=42,
        parsoff_data_path: Optional[str] = None,
        phate_data_path: Optional[str] = None,
        phicad_data_path: Optional[str] = None,
    ):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.parsoff_data_path = parsoff_data_path
        self.phate_data_path = phate_data_path
        self.phicad_data_path = phicad_data_path

    def load_english_dataset(self) -> pd.DataFrame:
        """Load the hate_speech_offensive dataset from HuggingFace"""
        logger.info("Loading English hate_speech_offensive dataset...")
        dataset = load_dataset("hate_speech_offensive", split="train")
        df = pd.DataFrame(dataset)

        # Map labels to our schema
        df["label_text"] = df["class"].map({0: "hateful", 1: "sensitive", 2: "safe"})
        df["text"] = df["tweet"]
        df["language"] = "en"
        df["source"] = "hate_speech_offensive"

        logger.info(f"Loaded {len(df)} English samples from hate_speech_offensive")
        return df[["text", "label_text", "language", "source"]]

    def load_parsoff_dataset(self) -> pd.DataFrame:
        """Load the Pars-OFF Farsi dataset"""
        if not self.parsoff_data_path:
            logger.warning("Pars-OFF data path not provided. Skipping Pars-OFF data.")
            return pd.DataFrame()

        logger.info("Loading Pars-OFF dataset...")

        # Load the main dataset file
        pars_off_path = Path(self.parsoff_data_path) / "Pars-OFF" / "Pars-OFF_all.csv"

        if not pars_off_path.exists():
            logger.error(f"Pars-OFF dataset not found at {pars_off_path}")
            return pd.DataFrame()

        # Read the CSV file
        df = pd.read_csv(pars_off_path)

        # Create a unified label based on the hierarchical structure
        df["label_text"] = df.apply(self._map_parsoff_labels, axis=1)
        df["text"] = df["tweet"]
        df["language"] = "fa"
        df["source"] = "pars_off"

        logger.info(f"Loaded {len(df)} Farsi samples from Pars-OFF")
        logger.info(f"Pars-OFF label distribution:\n{df['label_text'].value_counts()}")

        return df[["text", "label_text", "language", "source"]]

    def load_phate_dataset(self) -> pd.DataFrame:
        """Load the PHATE Persian dataset"""
        if not self.phate_data_path:
            logger.warning("PHATE data path not provided. Skipping PHATE data.")
            return pd.DataFrame()

        logger.info("Loading PHATE dataset...")

        # Path to PHATE data directory
        phate_path = Path(self.phate_data_path) / "Phate"

        # Try to load the pre-split files if they exist
        train_path = phate_path / "train_simple.csv"
        val_path = phate_path / "val_simple.csv"
        test_path = phate_path / "test_simple.csv"

        # Also check for the main dataset file
        main_path = phate_path / "Dataset_with_span_and_target.csv"

        dfs = []

        # Load pre-split files if they exist
        for file_path, split_name in [
            (train_path, "train"),
            (val_path, "val"),
            (test_path, "test"),
        ]:
            if file_path.exists():
                df = pd.read_csv(file_path)
                df["split"] = split_name
                dfs.append(df)
                logger.info(f"Loaded {len(df)} samples from PHATE {split_name}")

        # If pre-split files don't exist, try the main file
        if not dfs and main_path.exists():
            df = pd.read_csv(main_path)
            df["split"] = "full"
            dfs.append(df)
            logger.info(f"Loaded {len(df)} samples from PHATE main dataset")

        if not dfs:
            logger.error(f"No PHATE dataset files found in {phate_path}")
            return pd.DataFrame()

        # Combine all loaded dataframes
        combined_df = pd.concat(dfs, ignore_index=True)

        # Map PHATE labels to our unified schema
        combined_df["label_text"] = combined_df.apply(self._map_phate_labels, axis=1)
        combined_df["language"] = "fa"
        combined_df["source"] = "phate"

        # Keep the split information if available
        result_df = combined_df[["text", "label_text", "language", "source"]].copy()
        if "split" in combined_df.columns:
            result_df["phate_split"] = combined_df["split"]

        logger.info(f"Total PHATE samples loaded: {len(result_df)}")
        logger.info(
            f"PHATE label distribution:\n{result_df['label_text'].value_counts()}"
        )

        return result_df

    def load_phicad_dataset(self) -> pd.DataFrame:
        """Load the PHICAD Persian dataset, excluding spam"""
        if not self.phicad_data_path:
            logger.warning("PHICAD data path not provided. Skipping PHICAD data.")
            return pd.DataFrame()

        logger.info("Loading PHICAD dataset...")

        # Path to PHICAD data directory
        phicad_path = Path(self.phicad_data_path) / "PHICAD"

        part1_path = phicad_path / "PHICAD-part1.csv"
        part2_path = phicad_path / "PHICAD-part2.csv"

        dfs = []

        for file_path in [part1_path, part2_path]:
            if file_path.exists():
                df = pd.read_csv(file_path, sep="\t")
                dfs.append(df)
                logger.info(f"Loaded {len(df)} samples from {file_path.name}")
            else:
                logger.warning(f"PHICAD file not found: {file_path}")

        if not dfs:
            logger.error(f"No PHICAD dataset files found in {phicad_path}")
            return pd.DataFrame()

        # Combine parts
        combined_df = pd.concat(dfs, ignore_index=True)

        # Standardize text column
        combined_df["text"] = combined_df["comment_normalized"]

        # Filter out spam
        combined_df = combined_df[combined_df["spam"] != 1]

        # Map labels
        combined_df["label_text"] = combined_df.apply(
            lambda row: (
                "hateful"
                if row["hate"] == 1
                else "sensitive" if row["obscene"] == 1 else "safe"
            ),
            axis=1,
        )

        combined_df["language"] = "fa"
        combined_df["source"] = "phicad"

        result_df = combined_df[["text", "label_text", "language", "source"]]

        logger.info(f"Total PHICAD samples after filtering: {len(result_df)}")
        logger.info(
            f"PHICAD label distribution:\n{result_df['label_text'].value_counts()}"
        )

        return result_df

    def load_combined_farsi_dataset(self) -> pd.DataFrame:
        """
        Load and combine all Farsi datasets (Pars-OFF, PHATE, PHICAD) before any processing.
        This ensures balanced representation of all sources in the final dataset.
        """
        logger.info("\n" + "=" * 60)
        logger.info("LOADING COMBINED FARSI DATASETS (Pars-OFF + PHATE + PHICAD)")
        logger.info("=" * 60)

        farsi_datasets = []

        # Load Pars-OFF
        parsoff_df = self.load_parsoff_dataset()
        if not parsoff_df.empty:
            farsi_datasets.append(parsoff_df)
            logger.info(f"✓ Pars-OFF loaded: {len(parsoff_df)} samples")
        else:
            logger.warning("✗ Pars-OFF data not loaded")

        # Load PHATE
        phate_df = self.load_phate_dataset()
        if not phate_df.empty:
            farsi_datasets.append(phate_df)
            logger.info(f"✓ PHATE loaded: {len(phate_df)} samples")
        else:
            logger.warning("✗ PHATE data not loaded")

        # Load PHICAD
        phicad_df = self.load_phicad_dataset()
        if not phicad_df.empty:
            farsi_datasets.append(phicad_df)
            logger.info(f"✓ PHICAD loaded: {len(phicad_df)} samples")
        else:
            logger.warning("✗ PHICAD data not loaded")

        if not farsi_datasets:
            logger.warning("No Farsi datasets loaded!")
            return pd.DataFrame()

        # Combine all Farsi datasets
        combined_farsi_df = pd.concat(farsi_datasets, ignore_index=True)

        # Log combined statistics
        logger.info(f"\n{'='*60}")
        logger.info(f"COMBINED FARSI DATASET STATISTICS (Before Balancing)")
        logger.info(f"{'='*60}")
        logger.info(f"Total Farsi samples: {len(combined_farsi_df)}")
        logger.info(
            f"\nSource distribution:\n{combined_farsi_df['source'].value_counts()}"
        )
        logger.info(
            f"\nLabel distribution:\n{combined_farsi_df['label_text'].value_counts()}"
        )

        # Cross-tabulation of source and labels
        source_label_dist = pd.crosstab(
            combined_farsi_df["source"], combined_farsi_df["label_text"]
        )
        logger.info(f"\nLabel distribution by source:\n{source_label_dist}")

        return combined_farsi_df

    def _map_phate_labels(self, row) -> str:
        """Map PHATE dataset labels to our unified schema"""
        # Check Violence, Hate, and Vulgar columns for more nuanced classification
        if row.get("Hate", 0) == 1 or row.get("Violence", 0) == 1:
            return "hateful"

        if row.get("Vulgar", 0) == 1:
            return "sensitive"

        # If none of the above, it's safe
        return "safe"

    def _map_parsoff_labels(self, row) -> str:
        """Map Pars-OFF hierarchical labels to our unified schema"""
        # First check level_a (offensive or not)
        if row.get("level_a") == "NOT":
            return "safe"

        # If offensive, check level_b for targeting
        if pd.notna(row.get("level_b")):
            if row["level_b"] == "TIN":  # Targeted insult
                return "hateful"
            elif row["level_b"] == "UNT":  # Untargeted
                return "sensitive"

        # Default to sensitive if offensive but no clear targeting info
        return "sensitive"

    def load_all_datasets(self) -> pd.DataFrame:
        """Load and combine English and combined Farsi datasets"""
        datasets = []

        # Load English data
        english_df = self.load_english_dataset()
        if not english_df.empty:
            datasets.append(english_df)

        # Load combined Farsi data (Pars-OFF + PHATE + PHICAD together)
        farsi_df = self.load_combined_farsi_dataset()
        if not farsi_df.empty:
            datasets.append(farsi_df)

        if not datasets:
            raise ValueError("No datasets loaded. Please check data paths.")

        # Combine all datasets
        combined_df = pd.concat(datasets, ignore_index=True)

        # Clean NaN texts
        original_len = len(combined_df)
        combined_df = combined_df.dropna(subset=["text"])
        combined_df["text"] = combined_df["text"].astype(str)
        dropped = original_len - len(combined_df)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows with NaN or missing text")

        # Log statistics
        logger.info(f"\n{'='*60}")
        logger.info(f"FINAL COMBINED DATASET STATISTICS (All Languages)")
        logger.info(f"{'='*60}")
        logger.info(f"Total samples: {len(combined_df)}")
        logger.info(
            f"\nLanguage distribution:\n{combined_df['language'].value_counts()}"
        )
        logger.info(
            f"\nLabel distribution:\n{combined_df['label_text'].value_counts()}"
        )
        logger.info(f"\nSource distribution:\n{combined_df['source'].value_counts()}")

        # Cross-tabulation of language and labels
        lang_label_dist = pd.crosstab(
            combined_df["language"], combined_df["label_text"]
        )
        logger.info(f"\nLabel distribution by language:\n{lang_label_dist}")

        # Cross-tabulation of source and labels
        source_label_dist = pd.crosstab(
            combined_df["source"], combined_df["label_text"]
        )
        logger.info(f"\nLabel distribution by source:\n{source_label_dist}")

        return combined_df

    def prepare_multilingual_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create train/val/test splits with stratification by language and label"""

        # Check if PHATE data has pre-defined splits
        if "phate_split" in df.columns:
            logger.info("Note: PHATE pre-defined splits detected but will be ignored")
            logger.info("Creating unified splits for combined Farsi dataset")
            df = df.drop(columns=["phate_split"])

        # Create splits with stratification
        splits = self._create_splits(df)

        # Add numeric labels for model training
        label_to_id = {"safe": 0, "sensitive": 1, "hateful": 2}
        for split_name, split_df in splits.items():
            if not split_df.empty:
                split_df["label"] = split_df["label_text"].map(label_to_id)

                logger.info(f"\n{'='*40}")
                logger.info(f"{split_name.upper()} SPLIT STATISTICS")
                logger.info(f"{'='*40}")
                logger.info(f"Total samples: {len(split_df)}")
                logger.info(
                    f"Language distribution:\n{split_df['language'].value_counts()}"
                )
                logger.info(
                    f"Label distribution:\n{split_df['label_text'].value_counts()}"
                )
                logger.info(
                    f"Source distribution:\n{split_df['source'].value_counts()}"
                )

                # Per-language label distribution
                for lang in split_df["language"].unique():
                    lang_df = split_df[split_df["language"] == lang]
                    logger.info(f"\n{lang.upper()} label distribution in {split_name}:")
                    logger.info(f"{lang_df['label_text'].value_counts()}")

                # Per-source label distribution for Farsi
                farsi_sources = split_df[split_df["language"] == "fa"][
                    "source"
                ].unique()
                if len(farsi_sources) > 0:
                    logger.info(f"\nFarsi sources in {split_name}:")
                    for source in farsi_sources:
                        source_df = split_df[split_df["source"] == source]
                        logger.info(f"  {source}: {len(source_df)} samples")
                        logger.info(
                            f"  Label dist: {source_df['label_text'].value_counts().to_dict()}"
                        )

        return splits

    def _create_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Helper method to create train/val/test splits with stratification"""
        # Create a composite stratification key
        df = df.copy()
        df["strat_key"] = df["language"] + "_" + df["label_text"]

        # First split: train+val vs test
        X = df.drop(columns=["strat_key"]).values
        y = df["strat_key"].values

        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        # Second split: train vs val
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=y_temp,
        )

        # Recreate dataframes with proper column names
        columns = [col for col in df.columns if col != "strat_key"]

        splits = {
            "train": pd.DataFrame(X_train, columns=columns),
            "val": pd.DataFrame(X_val, columns=columns),
            "test": pd.DataFrame(X_test, columns=columns),
        }

        return splits

    def balance_multilingual_dataset(
        self, df: pd.DataFrame, strategy="undersample_per_language"
    ) -> pd.DataFrame:
        """
        Balance a multilingual dataset to handle class imbalance across different languages.

        This method provides multiple strategies to balance classes within each language
        independently, ensuring fair representation of all classes across all languages
        in the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing at least two columns:
            - 'language': Language identifier for each sample
            - 'label_text': Class label for each sample
        strategy : str, optional (default="undersample_per_language")
            Balancing strategy to apply. Options:
            - "undersample_per_language": Reduces majority classes to match minority class
            count within each language. This prevents data loss across languages but
            may discard useful samples within overrepresented classes.
            - "oversample_minority": Increases minority classes to match majority class
            count within each language using resampling with replacement. This preserves
            all original samples but may lead to overfitting on duplicated samples.
            - Any other value: Returns the original dataframe without balancing.

        Returns
        -------
        pd.DataFrame
            Balanced dataframe with the same columns as input, shuffled randomly.
            The returned dataframe maintains language separation during balancing
            and ensures equal class distribution within each language.
        """

        logger.info(f"\n{'='*60}")
        logger.info(f"BALANCING DATASET")
        logger.info(f"{'='*60}")
        logger.info(f"Strategy: '{strategy}'")
        logger.info(
            f"Original: {len(df)} samples across {df['language'].nunique()} languages"
        )

        # Log per-language statistics before balancing
        for lang in df["language"].unique():
            lang_df = df[df["language"] == lang]
            logger.info(f"\n{lang.upper()} before balancing:")
            logger.info(f"  Total: {len(lang_df)} samples")
            logger.info(
                f"  Label distribution:\n{lang_df['label_text'].value_counts()}"
            )
            if lang == "fa":
                logger.info(
                    f"  Source distribution:\n{lang_df['source'].value_counts()}"
                )

        if strategy == "undersample_per_language":
            balanced_dfs = []
            total_samples_removed = 0
            original_size = len(df)

            # Balance within each language
            for language in df["language"].unique():
                lang_df = df[df["language"] == language]
                min_class_count = lang_df["label_text"].value_counts().min()
                lang_original = len(lang_df)

                # Sample each class to match the minority class count
                for label in lang_df["label_text"].unique():
                    label_df = lang_df[lang_df["label_text"] == label]
                    target_count = min(min_class_count, len(label_df))
                    total_samples_removed += len(label_df) - target_count

                    sampled_df = label_df.sample(
                        n=target_count,
                        random_state=self.random_state,
                    )
                    balanced_dfs.append(sampled_df)

            # Combine all balanced language datasets and shuffle
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
            balanced_df = balanced_df.sample(
                frac=1, random_state=self.random_state
            ).reset_index(drop=True)

            reduction_pct = (total_samples_removed / original_size) * 100
            logger.info(
                f"\nUndersampling: Removed {total_samples_removed} samples ({reduction_pct:.1f}% reduction)"
            )

            # Warning threshold: if more than 30% of data is removed
            if reduction_pct > 30:
                logger.warning(
                    f"⚠️  SEVERE undersampling detected: {reduction_pct:.1f}% of data removed!"
                )
                logger.warning(
                    f"   Consider using 'oversample_minority' strategy or collecting more data"
                )

        elif strategy == "oversample_minority":
            from sklearn.utils import resample

            balanced_dfs = []
            total_samples_added = 0
            original_size = len(df)

            for language in df["language"].unique():
                lang_df = df[df["language"] == language]
                max_class_count = lang_df["label_text"].value_counts().max()

                # Resample each class to match the majority class count
                for label in lang_df["label_text"].unique():
                    label_df = lang_df[lang_df["label_text"] == label]
                    samples_added = max_class_count - len(label_df)
                    total_samples_added += samples_added

                    resampled_df = resample(
                        label_df,
                        n_samples=max_class_count,
                        random_state=self.random_state,
                    )
                    balanced_dfs.append(resampled_df)

            # Combine all balanced language datasets and shuffle
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
            balanced_df = balanced_df.sample(
                frac=1, random_state=self.random_state
            ).reset_index(drop=True)

            increase_pct = (total_samples_added / original_size) * 100
            duplication_rate = (total_samples_added / len(balanced_df)) * 100
            logger.info(
                f"\nOversampling: Added {total_samples_added} duplicate samples ({increase_pct:.1f}% increase)"
            )
            logger.info(f"Duplication rate: {duplication_rate:.1f}% of final dataset")

            # Warning threshold: if more than 50% of final dataset is duplicates
            if duplication_rate > 50:
                logger.warning(
                    f"⚠️  SEVERE oversampling detected: {duplication_rate:.1f}% of data is duplicates!"
                )
                logger.warning(
                    f"   High risk of overfitting - consider collecting more data or using augmentation"
                )

        else:
            logger.info(f"No balancing applied - returning original dataset")
            balanced_df = df

        # Log final statistics
        logger.info(f"\n{'='*60}")
        logger.info(f"AFTER BALANCING")
        logger.info(f"{'='*60}")
        logger.info(f"Balanced dataset: {len(balanced_df)} samples")
        logger.info(f"Language distribution:\n{balanced_df['language'].value_counts()}")
        logger.info(f"Label distribution:\n{balanced_df['label_text'].value_counts()}")

        # Per-language statistics after balancing
        for lang in balanced_df["language"].unique():
            lang_df = balanced_df[balanced_df["language"] == lang]
            logger.info(f"\n{lang.upper()} after balancing:")
            logger.info(f"  Total: {len(lang_df)} samples")
            logger.info(
                f"  Label distribution:\n{lang_df['label_text'].value_counts()}"
            )
            if lang == "fa":
                logger.info(
                    f"  Source distribution:\n{lang_df['source'].value_counts()}"
                )

        return balanced_df

    def save_splits(
        self, splits: Dict[str, pd.DataFrame], output_dir: str = "./data/processed"
    ) -> None:
        """
        Save train/val/test splits to CSV files.

        Parameters
        ----------
        splits : Dict[str, pd.DataFrame]
            Dictionary containing 'train', 'val', and 'test' dataframes
        output_dir : str, optional (default="./data/processed")
            Directory where the CSV files will be saved
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'='*60}")
        logger.info(f"SAVING DATASET SPLITS")
        logger.info(f"{'='*60}")
        logger.info(f"Output directory: {output_path.absolute()}")

        # Save each split
        saved_files = []
        for split_name, split_df in splits.items():
            if split_df.empty:
                logger.warning(f"⚠️  {split_name} split is empty, skipping save")
                continue

            # Define output file path
            output_file = output_path / f"{split_name}.csv"

            # Save to CSV
            split_df.to_csv(output_file, index=False, encoding="utf-8")

            # Log save information
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            logger.info(f"\n✓ Saved {split_name}.csv")
            logger.info(f"  - Path: {output_file.absolute()}")
            logger.info(f"  - Rows: {len(split_df):,}")
            logger.info(f"  - Columns: {len(split_df.columns)}")
            logger.info(f"  - Size: {file_size_mb:.2f} MB")
            logger.info(
                f"  - Languages: {split_df['language'].value_counts().to_dict()}"
            )
            logger.info(
                f"  - Labels: {split_df['label_text'].value_counts().to_dict()}"
            )

            saved_files.append(output_file)

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"SAVE COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total files saved: {len(saved_files)}")
        total_size_mb = sum(f.stat().st_size for f in saved_files) / (1024 * 1024)
        logger.info(f"Total size: {total_size_mb:.2f} MB")
        logger.info(f"\nSaved files:")
        for f in saved_files:
            logger.info(f"  - {f.name}")

    def get_dataset_statistics(self, df: pd.DataFrame) -> Dict:
        """Get comprehensive statistics about the multilingual dataset"""
        stats = {
            "total_samples": len(df),
            "languages": df["language"].value_counts().to_dict(),
            "labels": df["label_text"].value_counts().to_dict(),
            "sources": (
                df["source"].value_counts().to_dict() if "source" in df.columns else {}
            ),
            "cross_tabulation": pd.crosstab(df["language"], df["label_text"]).to_dict(),
        }

        # Text length statistics per language
        for lang in df["language"].unique():
            lang_df = df[df["language"] == lang]
            lengths = [len(str(text).split()) for text in lang_df["text"]]
            stats[f"{lang}_text_stats"] = {
                "avg_length": np.mean(lengths),
                "max_length": max(lengths),
                "min_length": min(lengths),
                "std_length": np.std(lengths),
            }

        # Statistics per source if available
        if "source" in df.columns:
            for source in df["source"].unique():
                source_df = df[df["source"] == source]
                stats[f"{source}_label_dist"] = (
                    source_df["label_text"].value_counts().to_dict()
                )

        return stats


# usage
if __name__ == "__main__":
    # Initialize the multilingual dataloader with all three datasets
    loader = MultilingualHateSpeechDataLoader(
        test_size=0.15,
        val_size=0.15,
        random_state=42,
        parsoff_data_path="./data",  # Path to directory containing Pars-OFF
        phate_data_path="./data",  # Path to directory containing Phate
        phicad_data_path="./data",  # Path to directory containing PHICAD
    )

    # Load all datasets (English + Combined Farsi [Pars-OFF + PHATE + PHICAD])
    combined_df = loader.load_all_datasets()

    # Create balanced dataset (now Farsi datasets are combined before balancing)
    balanced_df = loader.balance_multilingual_dataset(
        combined_df, strategy="undersample_per_language"
    )

    # Prepare splits
    splits = loader.prepare_multilingual_splits(balanced_df)

    # Save splits to CSV files
    loader.save_splits(splits, output_dir="./data/processed")

    # Get statistics
    stats = loader.get_dataset_statistics(combined_df)
    print("\n" + "=" * 60)
    print("FINAL DATASET STATISTICS")
    print("=" * 60)
    for key, value in stats.items():
        print(f"{key}: {value}")
