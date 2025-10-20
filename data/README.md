# Multilingual Hate Speech Detection: Data Preparation

This directory contains the scripts and documentation for preparing a multilingual hate speech dataset in English and Farsi. The goal is to create a unified and balanced dataset for training a robust hate speech detection model. This README provides a comprehensive overview of the datasets used, the data loading and cleaning procedures, and instructions on how to use the provided Python scripts.

The decision to use a combination of English and Farsi datasets was driven by the relative abundance of high-quality, large-scale English datasets and the smaller size of available Farsi datasets. By combining these resources, we aim to leverage the strengths of models trained on extensive English data while adapting and fine-tuning them for the nuances of the Farsi language.

## Table of Contents

- [Dataset Descriptions](#dataset-descriptions)
  - [Farsi Datasets](#farsi-datasets)
    - [Pars-OFF](#pars-off)
    - [PHICAD](#phicad)
    - [PHATE](#phate)
  - [English Dataset](#english-dataset)
    - [Hate Speech and Offensive Language](#hate-speech-and-offensive-language)
- [Data Loading and Preprocessing](#data-loading-and-preprocessing)
  - [Unified Labeling Scheme](#unified-labeling-scheme)
  - [Data Cleaning and Balancing](#data-cleaning-and-balancing)
- [How to Use the Code](#how-to-use-the-code)

## Dataset Descriptions

We utilized three Farsi datasets (Pars-OFF, PHICAD, and PHATE) and one English dataset.

### Farsi Datasets

#### Pars-OFF

*   **Description**: Pars-OFF is a benchmark dataset for offensive language detection in Farsi, collected from social media. It features a hierarchical annotation scheme with three levels, which allows for a nuanced understanding of offensive content.
*   **Original Labels**:
    *   `level_a`: `OFF` (offensive) vs. `NOT` (not offensive)
    *   `level_b`: `TIN` (targeted insult) vs. `UNT` (untargeted)
    *   `level_c`: Target category (e.g., individual, group)
*   **Our Cleaning Procedure**:
    *   We map the hierarchical labels to our unified schema:
        *   `NOT` in `level_a` is labeled as **safe**.
        *   `TIN` in `level_b` is labeled as **hateful**.
        *   `UNT` in `level_b` is labeled as **sensitive**.

#### PHICAD

*   **Description**: The Persian Hate Speech and Insult Corpus of Annotated Data (PHICAD) is a dataset of user comments from a Persian social network. It includes annotations for hate speech, obscene content, and spam.
*   **Original Labels**:
    *   `hate`: 1 if hate speech, 0 otherwise
    *   `obscene`: 1 if obscene, 0 otherwise
    *   `spam`: 1 if spam, 0 otherwise
*   **Our Cleaning Procedure**:
    *   Spam comments (where `spam` is 1) are removed from the dataset.
    *   The remaining data is mapped to our unified schema:
        *   `hate` = 1 is labeled as **hateful**.
        *   `obscene` = 1 (and `hate` = 0) is labeled as **sensitive**.
        *   Otherwise, the comment is labeled as **safe**.

#### PHATE

*   **Description**: PHATE is a Persian multi-label hate speech dataset containing over 7,000 manually annotated tweets. It also includes annotations for the target of the hate speech and the specific text spans that indicate hate speech, providing valuable context.
*   **Original Labels**: Multi-label annotations including `Hate`, `Violence`, and `Vulgar`.
*   **Our Cleaning Procedure**:
    *   The multi-label annotations are mapped to our unified schema:
        *   If `Hate` or `Violence` is present, it is labeled as **hateful**.
        *   If `Vulgar` is present (without `Hate` or `Violence`), it is labeled as **sensitive**.
        *   If none of these labels are present, it is labeled as **safe**.

### English Dataset

#### Hate Speech and Offensive Language

*   **Description**: This is a widely-used English dataset from a Kaggle competition, containing tweets annotated for hate speech, offensive language, or neither. It consists of over 24,000 tweets that were manually annotated.
*   **Original Labels**:
    *   0: Hate Speech
    *   1: Offensive Language
    *   2: Neither
*   **Our Cleaning Procedure**:
    *   The original labels are mapped to our unified schema:
        *   0 (Hate Speech) is labeled as **hateful**.
        *   1 (Offensive Language) is labeled as **sensitive**.
        *   2 (Neither) is labeled as **safe**.

## Data Loading and Preprocessing

The provided Python script (`MultilingualHateSpeechDataLoader`) handles the loading, cleaning, and preprocessing of these datasets to create a unified, balanced, and analysis-ready dataset.

### Unified Labeling Scheme

To create a consistent dataset for our multilingual model, we adopted the following three-tier labeling system:

*   **safe**: Content that is not offensive or hateful.
*   **sensitive**: Content that is offensive but not directed at a specific group or individual (e.g., general profanity).
*   **hateful**: Content that is abusive and targets a specific group or individual based on characteristics such as race, religion, gender, etc.

### Data Cleaning and Balancing

The script performs the following key preprocessing steps:

1.  **Loading and Combining**: Loads each of the four datasets and combines them into a single pandas DataFrame.
2.  **Label Mapping**: Converts the original labels of each dataset into our unified three-tier schema.
3.  **Text Cleaning**: Removes any rows with missing or empty text.
4.  **Data Balancing**: To address the class imbalance, an undersampling strategy is applied *per language*. This ensures that each class (`safe`, `sensitive`, `hateful`) has an equal number of samples within both the English and Farsi subsets of the data. This prevents the model from being biased towards the majority class.
5.  **Data Splitting**: The balanced dataset is then split into training, validation, and test sets with stratification on both language and label to ensure that the distribution of data is consistent across all splits.

## How to Use the Code

1.  **Prerequisites**: Ensure you have Python 3.x and the required libraries installed:
    ```bash
    pip install pandas scikit-learn datasets
    ```
2.  **Data Setup**:
    *   Place the Farsi datasets (`Pars-OFF`, `Phate`, and `PHICAD`) in a directory named `data`.
    *   The English `hate_speech_offensive` dataset will be downloaded automatically from the Hugging Face Hub.
3.  **Run the Script**: Execute the main script to process the data:
    ```bash
    python -m data.data_loader
    ```
    The script will output the processed and split datasets into the `./data/processed` directory as `train.csv`, `val.csv`, and `test.csv`.

