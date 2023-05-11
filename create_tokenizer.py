import functools
from transformers import AutoTokenizer
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


@functools.cache
def load_spacy_nlp():
    return spacy.load("ru_core_news_lg")


def get_new_tokens(text):
    nlp = load_spacy_nlp()
    doc = nlp(text)
    tokens = [
        token.text
        for token in doc
        if (
            not token.is_stop
            and not token.is_punct
            and token.text.strip() != ""
            and token.text.find("\n") == -1
        )
    ]
    return tokens


def add_actor_tokens(tokenizer):
    nlp = load_spacy_nlp()
    df = pd.read_csv("./data/columns.csv", sep="<")
    actor_names = df[df["label"] == "актёр"]["column"].astype(str).to_list()
    nlp.max_length = 10000000
    tfidf_vectorizer = TfidfVectorizer(lowercase=False, tokenizer=get_new_tokens)
    tfidf_vectorizer.fit_transform(actor_names[:3000])
    idf = tfidf_vectorizer.idf_
    idf_sorted_indexes = sorted(range(len(idf)), key=lambda k: idf[k])
    tokens_by_df = np.array(tfidf_vectorizer.get_feature_names_out())[
        idf_sorted_indexes
    ]
    tokenizer.add_tokens(list(tokens_by_df[:2000]))

    return tokenizer


def build_wider_tokenizer(pretrained_model_name):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    initial_tokens_number = len(tokenizer)

    tokenizer = add_actor_tokens(tokenizer)

    new_tokens = []
    df = pd.read_csv("./data/columns.csv", sep="<")
    labels = ["вид спорта", "столица", "жанр"]
    for label in labels:
        new_tokens += get_new_tokens(
            " ".join(df[df["label"] == label]["column"].astype(str).to_list())
        )

    tokenizer.add_tokens(new_tokens)

    print(f"Added {len(tokenizer) - initial_tokens_number} of new tokens")

    return tokenizer


if __name__ == "__main__":
    pretrained_model_name = "cointegrated/rubert-tiny2"
    tokenizer = build_wider_tokenizer(pretrained_model_name)
    tokenizer.save_pretrained("rudoduo_tokenizer")
