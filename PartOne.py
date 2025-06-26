#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import spacy
from pathlib import Path
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import cmudict
import pickle 
from collections import Counter

#nltk.download('cmudict')
#nltk.download("punkt")
#nltk.download("punkt_tab") 
#nltk.download("averaged_perceptron_tagger")
#nltk.download("cmudict")

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000

# Part a
def read_novels(path=Path.cwd() / "novels"):
    """
    1. Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year
    2. sort the dataframe by the year column before returning it, resetting or ignoring the dataframe index.
    """
    if not path.is_dir():
        print("Error: Directory not found")

    data = [] # The list of dictionaries where i will put the data of each book, each novel will be a dict
    for novel in path.glob("*.txt"):
        name = novel.name # Sense_and_Sensibility-Austen-1811.txt
        title, author, year = name.split("-")
        year = int(year[:4])

        # Open the txt
        with open(novel, encoding = "utf-8") as f:
            text = f.read()

        dic = {"text": text, "author": author, "title": title, "year": year}
        data.append(dic)

    # Once we have the list of dict, we have to create the data frame
    df = pd.DataFrame.from_records(data)
    df = df.sort_values("year")
    df = df.reset_index()
    return df

# Part b
def nltk_ttr(text):
    """
    Calculates the type-token ratio of a text. 
    Text is tokenized using nltk.word_tokenize.
    """
    text = text.lower()
    tokens = nltk.word_tokenize(text) 
    tokens = [t for t in tokens if t.isalpha()] # Take only the alfabetical tokens 
    types = set(tokens)
    return len(types) / len(tokens) if tokens else 0

# Part c
def flesch_kincaid(df):
    """
    This function should return a dictionary mapping the title of
    each novel to the Flesch-Kincaid reading grade level score of the text. Use the
    NLTK library for tokenization and the CMU pronouncing dictionary for esti-
    mating syllable counts.
    """
    d = {}
    titles = list(df["title"])
    for title in titles:
        text = df[df["title"]== title]["text"].values[0]
        text = text.lower()

        # Number of words
        tokens = nltk.word_tokenize(text) 
        tokens = [t for t in tokens if t.isalpha()] # Take only the alfabetical tokens 
        n_words = len(tokens)

        # Number of sentences
        sent = nltk.sent_tokenize(text)
        n_sent = len(sent)

        # Aproximation of sylabels
        dict = cmudict.dict()
        n_syl = 0
        for token in tokens:
            if token in dict:
                phonemes = dict[token] #List of lists, posible phonemes
                # I am going to pick the longest element of the list, i rather overestimate the difficulty
                a = len(phonemes[0])
                phonem = phonemes[0]
                for p in phonemes:
                    if len(p) > a:
                        phonem = p
                        a = len(p)
                n_syl += len(phonem)
            else:
                n_syl += 1
    
        # Calculate Flesch-Kincaid
        FKGL = 0.39 *(n_words/n_sent)+ 11.8 *(n_syl/n_words)-15.59

        d[title] = FKGL # Add to the dictionary
    return d

def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for _, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results

def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.
    """
    pass

# Part e
def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """
    Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file
    """
    PATH = Path("/parsed_novels.pkl")
    
    if PATH.exists():
        """
        iv. Load the dataframe from the pickle file and use it for the remainder of this
        oursework part.
        """
        with open(PATH, "rb") as f:
            return pickle.load(f)
    
    else:
        """
        i. Use the spaCy nlp method to add a new column to the dataframe that
        contains parsed and tokenized Doc objects for each text.
        """
        nlp.max_length = 2000000
        parses = []
        texts = list(df["text"])
        for text in texts:
            parse = nlp(text)
            parses.append(parse)
        df["parse"] = parses

        """
        ii. Serialise the resulting dataframe (i.e., write it out to disk) using the pickle
        format.
        """
        # I am saving it at my computer bc i had some problems with github 
        # when i had the document in this file
        with open(PATH, "wb") as f:
            pickle.dump(df, f)

        """
        iii. Return the dataframe.
        """
        return df

# Part f
# Part f.i 
def most_common_objects(df):
    """
    The title of each novel and a list of the ten most common syntactic objects
    overall in the text. (No esta acabada)
    """
    d = {}
    titles = list(df["title"])
    for title in titles:
        parse = df[df["title"]== title]["parse"]
        for token in parse:
            print(token.text)
            print(type(token))
        d[title] = token
    print(d)
    pass 

def adjective_counts(df, n = 5):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    d = {}
    adjectives = []
    for index, row in df.iterrows():
        doc = row["parse"]
        for token in doc:
            if token.pos_ == "ADJ":
                adjectives.append(token.text.lower() )
        top_adj = Counter(adjectives).most_common(n)

        d[row["title"]] = top_adj
    return d

def subjects_by_verb_count(df, verb, n=5):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    subjects = []
    for token in df:
        if token.pos_ == "VERB" and token.lemma_.lower() == verb.lower():
            for child in token.children:
                # token.children is a list of words in the sentence that are related to the token
                if child.dep_ == "nsubj": # check if child is a subject 
                        subjects.append(child.text.lower() )
    d = Counter(subjects).most_common(n)                  
    return d


if __name__ == "__main__":
    
    #uncomment the following lines to run the functions once you have completed them
    
    path = Path.cwd() / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    print(df.head())
    nltk.download("cmudict")
    df = parse(df)
    print(df.head())
    print(flesch_kincaid(df))
    #print(get_ttrs(df))
    #print(get_fks(df))
    PATH = Path("/parsed_novels.pkl")
    df = pd.read_pickle(PATH)
    print(adjective_counts(df))
    
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parse"], "hear"))
        print("\n")
    """
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    """
