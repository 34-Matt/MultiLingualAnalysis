import os
import wget
from glob import glob
import tarfile
import pandas as pd

from typing import List

def getTatoeba() -> str:
    """
    Downloads the Tatoeba dataset.
    https://tatoeba.org/en/

    Returns:
        filepath (str): Path to the downloaded TSV file.
    """

    # Download
    if not os.path.exists("Downloads"):
        os.makedirs("Downloads")

    ## Sentence dataset
    save_bz2_path = os.path.join("Downloads", "sentences.tar.bz2")
    if not os.path.exists(save_bz2_path):
        print("Downloading Tatoeba dataset...")
        url = "https://downloads.tatoeba.org/exports/sentences.tar.bz2"
        wget.download(url, out = save_bz2_path)
    
    ## Pair dataset
    base_bz2_path = os.path.join("Downloads", "sentences_base.tar.bz2")
    if not os.path.exists(base_bz2_path):
        print("Downloading Tatoeba pairings...")
        url = "https://downloads.tatoeba.org/exports/sentences_base.tar.bz2"
        wget.download(url, out = base_bz2_path)

    return save_bz2_path, base_bz2_path

def __extractTatoeSentence() -> pd.DataFrame:
    """Extracts the Tatoeba dataset from a tar.bz2 file.
    
    Args:
        path (str): Path to the tar.bz2 file.
        cols (List[str]): List of column names for the DataFrame.

    Returns:
        data (pd.DataFrame): DataFrame containing the extracted data.
    """
    with tarfile.open(os.path.join("Downloads", "sentences.tar.bz2"), "r:bz2") as tar:
        member = tar.getmember("sentences.csv")
        file_obj = tar.extractfile(member)
        df = pd.read_csv(file_obj, delimiter='\t', header=None, names=["id", "lang", "text"])
    return df

def __extractTatoePairing() -> pd.DataFrame:
    with tarfile.open(os.path.join("Downloads", "sentences_base.tar.bz2"), "r:bz2") as tar:
        member = tar.getmember("sentences_base.csv")
        file_obj = tar.extractfile(member)
        df = pd.read_csv(file_obj, delimiter='\t', header=None, names=["id", "pair_id"])
    return df

def createTatoebaPairs(primary_language: str, secondary_languages: List[str]) -> pd.DataFrame:
    """
    Creates a DataFrame of sentence pairs from the Tatoeba dataset based on a primarly language.

    Args:
        primary_language (str): The primary language code.
        secondary_languages (List[str]): List of secondary language codes.

    Returns:
        pd.DataFrame: DataFrame containing sentence pairs.
    """
    # Load the Tatoeba dataset
    getTatoeba()
    sentence_df = __extractTatoeSentence()
    pairing_df = __extractTatoePairing()

    # Get all sentences in the primary language
    primary_sentences = sentence_df[sentence_df["lang"] == primary_language].copy()
    for lang in secondary_languages:
        primary_sentences[lang] = None

    # Filter pairing based on id of primary sentences
    pairing_df = pairing_df[pairing_df["pair_id"].isin(primary_sentences["id"])].copy()
    
    for pairing_series in pairing_df.iterrows():
        cur_lang = sentence_df[sentence_df["id"] == pairing_series[1]["pair_id"]]["lang"].values[0]
        if cur_lang in secondary_languages:
            primary_sentences.loc[primary_sentences["id"] == pairing_series[1]["pair_id"], cur_lang] = sentence_df[sentence_df["id"] == pairing_series[1]["pair_id"]]["text"].values[0]

    # Drop rows where all secondary languages are None
    primary_sentences = primary_sentences.dropna(subset=secondary_languages, how='all')
    primary_sentences = primary_sentences.reset_index(drop=True)
    primary_sentences = primary_sentences.drop(columns=["id", "lang"])

    return primary_sentences

def loadTatoebaPairs(lang1:str, lang2:str) -> pd.DataFrame:
    """Loads a paired dataset from the Tatoeba dataset.
    Requires downloading the dataset first. From https://tatoeba.org/en/downloads, download the sentence pairs.
    
    Args:
        lang1 (str): The first language code.
        lang2 (str): The second language code.

    Returns:
        Data (DataFrame): DataFrame containing the paired sentences.
    """
    filename_base = "Sentence pairs in {}-{}*.tsv"
    filename = glob(os.path.join("Downloads", filename_base.format(lang1, lang2)))
    if len(filename) == 0:
        filename = glob(filename_base.format(lang2, lang1))
    if len(filename) == 0:
        raise FileNotFoundError(f"No paired dataset found for languages {lang1} and {lang2}. Please download from https://tatoeba.org/en/downloads.")

    filename = filename[0]
    return pd.read_csv(filename, sep="\t", header=None, names=["id1", "lang1", "id2", "lang2"])

def loadMyWordPairs(lang: str) -> pd.DataFrame:
    """Loads a custom word pairs dataset.
    
    Args:
        lang (str): The language code.
    Returns:
        Data (DataFrame): DataFrame containing the word pairs.
    """
    if lang.lower() in ['japanese', 'jpn']:
        filename = os.path.join("Downloads", "JapaneseWords.csv")
        data = pd.read_csv(filename, header=0, names=["lang2", "lang1"], encoding='cp1252')
        data = data.iloc[:120, :]
        return data
    elif lang.lower() in ['spanish', 'spa']:
        filename = os.path.join("Downloads", "SpanishWords.csv")
        data = pd.read_csv(filename, header=0, names=["lang2", "lang1"], encoding='cp1252')
        data = data.iloc[:400, :]
        return data


if __name__ == "__main__":
    # Example usage
    primary_language = "eng"
    secondary_languages = ["spa", "jpn"]
    df = createTatoebaPairs(primary_language, secondary_languages)
    print(df.head())
    df.to_csv(os.path.join("Downloads", "processed_eng_spa_jpn.csv"), index=False)