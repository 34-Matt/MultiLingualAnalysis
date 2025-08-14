import os
import wget
from glob import glob
import tarfile
import pandas as pd
import json

from typing import List, Dict


EXAMPLE_TRANSLATION_TEXT = {
    "English2Japanese": {
        "English": "Translate this into Japanese.\nEnglish: {}\nJapanese: ",
        "Spanish": "Por favor traduzca al japonés.\nInglés: {}\nJaponés: ",
        "Japanese": "日本語に翻訳してください。\n英語:{}\n日本語:",
    },
    "Japanese2English": {
        "English": "Translate this into English.\nJapanese: {}\nEnglish: ",
        "Spanish": "Por favor traduzca al inglés.\nJaponés: {}\nInglés: ",
        "Japanese": "英語に翻訳してください。\n日本語:{}\n英語:",
    },
    "English2Spanish": {
        "English": "Translate this into Spanish.\nEnglish: {}\nSpanish: ",
        "Spanish": "Por favor traduzca al español.\nInglés: {}\nEspañol: ",
        "Japanese": "スペイン語に翻訳してください。\n英語:{}\nスペイン語:",
    },
    "Spanish2English": {
        "English": "Translate this into English.\nSpanish: {}\nEnglish: ",
        "Spanish": "Por favor traduzca al inglés.\nEspañol: {}\nInglés: ",
        "Japanese": "英語に翻訳してください。\nスペイン語:{}\n英語:",
    }
}

LANGUAGE_LONGHAND = {
    "English": ["english", "eng"],
    "Japanese": ["japanese", "jpn", "jp"],
    "Spanish": ["spanish", "esp", "sp"],
}

LANGUAGE_SHORTHAND = {
    "eng": ["english", "eng"],
    "jpn": ["japanese", "jpn", "jp"],
    "esp": ["spanish", "esp", "sp"],
}

def getLanguage(text: str, shorthand: bool = False) -> str:
    """Ensures a uniform method of refering to languages.
    
    Args:
        text (str): The input text referencing the languages
        shorthand (bool): Whether to return the shorthand (True) or longhand (False, default)

    Returns:
        output (str): The output text of the language
    """
    t = text.lower()
    if shorthand:
        dictionary = LANGUAGE_SHORTHAND
    else:
        dictionary = LANGUAGE_LONGHAND

    for key, value in dictionary.items():
        if t in value:
            return key
    raise IndexError(f"The language {text} is not found in the LANGUAGE_SHORTHAND dataset.")

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
    return pd.read_csv(filename, sep="\t", header=None, names=["id1", "lang1", "id2", "lang2"], on_bad_lines="warn")

def loadRedditPairs(lang: str) -> pd.DataFrame:
    """Loads the Reddit word pairs dataset.
    
    Args:
        lang (str): The language code.
    Returns:
        Data (DataFrame): DataFrame containing the word pairs.
    """
    if lang.lower() in LANGUAGE_SHORTHAND['jpn']:
        filename = os.path.join("Downloads", "JapaneseReddit.csv")
        data = pd.read_csv(filename, header=0, encoding='utf-8')
        data = data[["japanese", "translation"]]
        data.columns = ["lang2", "lang1"]
        data = data.dropna()
    elif lang.lower() in LANGUAGE_SHORTHAND['esp']:
        filename = os.path.join("Downloads", "SpanishReddit.csv")
        data = pd.read_csv(filename, header=None, names=["lang1", "lang2"], encoding='utf-8', on_bad_lines="warn") 
    else:
        raise ValueError(f"Unknown file for language {lang}")
    
    return data
    
def createPatternlenJson(
        unformatted_prompt: Dict[str, List[str]],
        text_input: pd.Series,
        text_target: pd.Series,
        output: str = "output.jsonl",
        append: bool = True,
        source: str = "personal",
        direction: str = "personal"
        ) -> None:
    """Creates a json file use with pattern lens.
    
    Args:
        unformatted_prompt (Dict[str, List[str]]): All the base strings used to generate the dataset. Unformatted string with one inputs for text. Keys are the title for the prompt.
        text_input (pd.Series): The text that goes into the unformatted_prompt
        text_target (pd.Series): The text that should be returned by the model when given the formatted string
        output (str): The output json file to save the formated prompts to (default to output.jsonl)
        append (bool): Whether to append the generated prompts to an existing json file (True; default) or not (False)
        source (str): Name of the source of the data (optional for tracking purposes)
        direction (str): The direction from starting language to target language (optional for tracking purposes)
    """
    mode = 'a' if append else 'w'
    with open(output, mode, encoding='utf-8') as f:
        for title, prompt in unformatted_prompt.items():
            for ti, tt in zip(text_input, text_target):
                formatted_prompt = prompt.format(ti)
                entry = {
                    "text": formatted_prompt,
                    "target": tt, 
                    "meta": {"pile_set_name": source, "direction": direction, "prompt": title}
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def loadPatternlenJson(file_name: str) -> pd.DataFrame:
    """Loads data from a jsonl file as required by patternlen
    
    Args:
        file_name (str): The file to load
    
    Returns:
        data (pd.DataFrame): The loaded data
    """
    return pd.read_json(file_name, lines=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download or process language datasets.")
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=["tatoeba", "reddit"],
        default=['reddit'],
        help="Which dataset to download/process: 'tatoeba', 'reddit' (default: tatoeba reddit)"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=["esp", "jpn"],
        default=["esp", "jpn"],
        help="Secondary language codes for Tatoeba (default: esp jpn)"
    )
    parser.add_argument(
        "--translate_sentence",
        action="store_true",
        help="When active, convert preloaded datasets into sentences for translations"
    )
    args = parser.parse_args()

    if args.translate_sentence:
        for lang in args.languages:
            l = getLanguage(lang)
            existing_file = False

            # Load the dataset
            for dataset in args.dataset:
                if dataset.lower() == "tatoeba":
                    cur_data = loadTatoebaPairs("English", l)
                elif dataset.lower() == "reddit":
                    cur_data = loadRedditPairs(l)
                else:
                    print(f"Cannot load dataset {dataset}")
                    continue
                try:
                    pass
                except Exception as e:
                    print(f"Found error will loading dataset {dataset}")
                    print(e)
                    continue
                
                # Create and save the prompts
                if l == "Japanese":
                    createPatternlenJson(
                        EXAMPLE_TRANSLATION_TEXT["English2Japanese"],
                        cur_data['lang1'],
                        cur_data['lang2'],
                        os.path.join("Downloads", "English2JapaneseTranslate.csv"),
                        existing_file,
                        dataset.title(),
                        "English2Japanese"
                    )
                    createPatternlenJson(
                        EXAMPLE_TRANSLATION_TEXT["Japanese2English"],
                        cur_data['lang2'],
                        cur_data['lang1'],
                        os.path.join("Downloads", "Japanese2EnglishTranslate.csv"),
                        existing_file,
                        dataset.title(),
                        "Japanese2English"
                    )
                elif l == "Spanish":
                    createPatternlenJson(
                        EXAMPLE_TRANSLATION_TEXT["English2Spanish"],
                        cur_data['lang1'],
                        cur_data['lang2'],
                        os.path.join("Downloads", "English2SpanishTranslate.csv"),
                        existing_file,
                        dataset.title(),
                        "English2Spanish"
                    )
                    createPatternlenJson(
                        EXAMPLE_TRANSLATION_TEXT["Spanish2English"],
                        cur_data['lang2'],
                        cur_data['lang1'],
                        os.path.join("Downloads", "Spanish2EnglishTranslate.csv"),
                        existing_file,
                        dataset.title(),
                        "Spanish2English"
                    )
                else:
                    raise IndexError(f"Unknown language {l} somehow overwrote `l`")
            
                existing_file = True

    else:
        if "tatoeba" in args.dataset:
            df = createTatoebaPairs('eng', args.languages)
            print(df.head())
            df.to_csv(
                os.path.join(
                    "Downloads",
                    f"processed_{args.primary_language}_{'_'.join(args.languages)}.csv"
                ),
                index=False
            )

        if "reddit" in args.dataset:
            for lang in args.languages:
                if lang.lower() in LANGUAGE_SHORTHAND["jpn"]:
                    print("Download the file from [u/Lammmas' Google Sheet](https://docs.google.com/spreadsheets/d/1cT16lcMnSoWW_VNO8DgPMKhPkXVj41ow7RZ0kuZQ4Jk/edit?gid=189116820#gid=189116820) and save it as `JapaneseReddit.csv` in the `Downloads` directory.")
                elif lang.lower() in LANGUAGE_SHORTHAND["esp"]:
                    print("Download the file from [u/raitro's Google Doc](https://docs.google.com/document/d/1aXm7BP-AYg3gU-4NWcGBjg-I8gd2fu6A2IUa8cvfV8Q/edit?tab=t.0). This must first be downloaded as a plain text file (`.txt`). Then, the file can be renamed to `SpanishReddit.csv` and saved in the `Downloads` directory.")
                else:
                    print(f"Unable to work with language: {lang}")
