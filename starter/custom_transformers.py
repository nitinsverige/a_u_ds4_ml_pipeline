"""Custom transformers (CT) using scikit-learn for the StyleSense pipeline.

Saved in .py file so that fitted pipeline can be picked using joblib library
and loaded later.

Part of speech (POS)

Function log1p_clipped  log(1 + x) safe for inputs
CT JoinTextColumns merges text columns (Title, Review Text) into one
CT CountCharacter  counts how often a given character appears in a text
CT SpacyLemmatizer lemmatizes text with spaCy
CT SpacyPosRatios proportion of adjectives, adverbs, verbs,.. in a text
"""
from __future__ import annotations

import numpy as np
import pandas as pd


from sklearn.base import BaseEstimator, TransformerMixin

DEFAULT_SPACY_MODEL = "en_core_web_sm"

# POS tags whose share of the review - used as a feature
POS_TAGS = ("NOUN", "ADJ", "ADV", "VERB",  "PRON")

# Stop words that we keep: negations and intensifiers flip or
DEFAULT_KEEP_WORDS = ("not", "no", "never", "nor", "none", "nothing",
                      "neither", "nobody", "too", "very", "really",
                      "only", "but", "however", "enough")

_CHARACTER_LABELS = {" ": "space", "!": "exclamation", "?": "question_mark"}
_MAX_CACHED_TEXTS = 200_000

# Functions


def log1p_clipped(values):
    """log(1 + x) : clipping """
    return np.log1p(np.clip(values, 0, None))


def _as_text_list(texts) -> list[str]:
    return [text if isinstance(text, str) else "" for text in texts]


_SPACY_MODELS: dict = {}
_ANALYSIS_CACHE: dict = {}


def _load_spacy(model_name: str):
    """Load once per process spaCy model
    excluding parser, NER."""
    if model_name not in _SPACY_MODELS:
        import spacy
        _SPACY_MODELS[model_name] = spacy.load(
            model_name, exclude=["parser", "ner"])
    return _SPACY_MODELS[model_name]


def analyze_texts(texts, model_name: str = DEFAULT_SPACY_MODEL) -> list[tuple]:
    """Run spaCy once per distinct text and remember result
    Cross-validation sees every review
    Parsing (slow): each text is parsed only once/process & result reused
    The cache holds plain Python data and is not part of the pipeline object,
    so it never ends up inside pickled model.
    Returns one <lemmas, is_stop,pos_ratios> tuple per input text
    """
    texts = _as_text_list(texts)
    cache = _ANALYSIS_CACHE.setdefault(model_name, {})
    if len(cache) > _MAX_CACHED_TEXTS:  # keep memory bound
        cache.clear()
    missing = [text for text in dict.fromkeys(texts) if text not in cache]
    if missing:
        nlp = _load_spacy(model_name)
        for text, doc in zip(missing, nlp.pipe(missing, batch_size=256)):
            tokens = [
                tok for tok in doc
                if not (tok.is_space or tok.is_punct)
            ]
            n_tokens = max(len(tokens), 1)
            lemmas = tuple(tok.lemma_.lower() for tok in tokens)
            is_stop = tuple(tok.is_stop for tok in tokens)
            pos_ratios = tuple(sum(tok.pos_ == tag for tok in tokens)/n_tokens
                               for tag in POS_TAGS)
            cache[text] = (lemmas, is_stop, pos_ratios)
    return [cache[text] for text in texts]


# custom transformers
class JoinTextColumns(BaseEstimator, TransformerMixin):
    """Join several text columns into a single text per row  """
    def __init__(self, separator: str = ". "):
        self.separator = separator

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).fillna("").astype(str)
        joined = frame.iloc[:, 0]
        for column in range(1, frame.shape[1]):
            joined = joined + self.separator + frame.iloc[:, column]
        return joined.to_numpy(dtype=object)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(["joined_text"], dtype=object)


class CountCharacter(BaseEstimator, TransformerMixin):
    """Count no of times character appears in each text"""
    def __init__(self, character: str):
        self.character = character

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array(
            [[text.count(self.character)] for text in _as_text_list(X)],
            dtype=float,)

    def get_feature_names_out(self, input_features=None):
        label = _CHARACTER_LABELS.get(self.character, self.character)
        return np.asarray([f"count_of_{label}"], dtype=object)


class SpacyPosRatios(BaseEstimator, TransformerMixin):
    """Proportion of tokens per POS tag"""
    def __init__(self, model_name: str = DEFAULT_SPACY_MODEL):
        self.model_name = model_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array(
            [ratios for _, _, ratios in analyze_texts(X, self.model_name)],
            dtype=float,)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(
            [f"share_of_{tag.lower()}" for tag in POS_TAGS], dtype=object)


class SpacyLemmatizer(BaseEstimator, TransformerMixin):
    """Lemmatize text with spaCy & return 1 normalized string per text """
    def __init__(self,
                 model_name: str = DEFAULT_SPACY_MODEL,
                 remove_stop_words: bool = True,
                 keep_words: tuple = DEFAULT_KEEP_WORDS):
        self.model_name = model_name
        self.remove_stop_words = remove_stop_words
        self.keep_words = keep_words

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        keep = set(self.keep_words)
        normalized = []
        for lemmas, is_stop, _ in analyze_texts(X, self.model_name):
            if self.remove_stop_words:
                lemmas = [
                    lemma
                    for lemma, stop in zip(lemmas, is_stop)
                    if not stop or lemma in keep]
            normalized.append(" ".join(lemmas))
        return normalized

    def get_feature_names_out(self, input_features=None):
        return np.asarray(["lemmatized_text"], dtype=object)
