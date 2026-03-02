"""Character n-gram tokenizer and feature extraction.

Direct parallel to k-mer extraction from genomics — character n-grams
capture local sequence patterns in DNS queries, system call traces,
or any character-level data.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Low-level n-gram extraction ─────────────────────────────────────


def extract_ngrams(text: str, n: int) -> list[str]:
    """Extract all character n-grams of length *n* from *text*.

    Analogous to k-mer extraction: ``extract_ngrams("abcde", 3)``
    returns ``["abc", "bcd", "cde"]``.
    """
    if n < 1 or len(text) < n:
        return []
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def extract_ngrams_range(text: str, n_min: int, n_max: int) -> list[str]:
    """Extract character n-grams for all sizes in ``[n_min, n_max]``."""
    ngrams: list[str] = []
    for n in range(n_min, n_max + 1):
        ngrams.extend(extract_ngrams(text, n))
    return ngrams


def ngram_frequency(text: str, n: int) -> dict[str, int]:
    """Return a frequency counter of character n-grams."""
    return dict(Counter(extract_ngrams(text, n)))


# ── Preprocessing ───────────────────────────────────────────────────


def preprocess_domain(domain: str) -> str:
    """Normalise a domain string for n-gram extraction.

    * Lower-case
    * Strip protocol prefix (``http(s)://``)
    * Strip trailing dot / whitespace
    * Replace dots with spaces (so n-grams don't span labels)
    """
    domain = domain.strip().lower()
    domain = re.sub(r"^https?://", "", domain)
    domain = domain.rstrip(".")
    # Replace dots with space so n-grams stay within labels
    domain = domain.replace(".", " ")
    return domain


# ── TF-IDF vectoriser wrapper ──────────────────────────────────────


class NgramTokenizer:
    """Character n-gram TF-IDF vectoriser (sklearn wrapper).

    Mirrors the k-mer frequency table approach used in genomics but
    produces TF-IDF weighted sparse feature matrices suitable for
    downstream classifiers.

    Parameters
    ----------
    ngram_range : tuple[int, int]
        Min and max n-gram sizes, e.g. ``(2, 4)``.
    max_features : int
        Maximum vocabulary size (top features by TF-IDF).
    min_df, max_df : int | float
        Minimum / maximum document-frequency thresholds.
    sublinear_tf : bool
        Apply sublinear (log) TF scaling.
    """

    def __init__(
        self,
        ngram_range: tuple[int, int] = (2, 4),
        max_features: int = 5000,
        min_df: int | float = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
    ) -> None:
        self.ngram_range = ngram_range
        self.max_features = max_features
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
        )
        self._is_fitted = False

    # ── Fit / Transform ─────────────────────────────────────────────

    def fit(self, texts: Sequence[str]) -> NgramTokenizer:
        """Fit the TF-IDF vocabulary on a corpus of raw domain strings."""
        processed = [preprocess_domain(t) for t in texts]
        self._vectorizer.fit(processed)
        self._is_fitted = True
        return self

    def transform(self, texts: Sequence[str]) -> csr_matrix:
        """Transform raw domain strings to TF-IDF feature matrix."""
        if not self._is_fitted:
            raise RuntimeError("NgramTokenizer has not been fitted yet.")
        processed = [preprocess_domain(t) for t in texts]
        return self._vectorizer.transform(processed)

    def fit_transform(self, texts: Sequence[str]) -> csr_matrix:
        """Fit and transform in one step."""
        processed = [preprocess_domain(t) for t in texts]
        mat = self._vectorizer.fit_transform(processed)
        self._is_fitted = True
        return mat

    # ── Introspection ───────────────────────────────────────────────

    @property
    def vocabulary(self) -> dict[str, int]:
        """Return the fitted vocabulary mapping n-gram → index."""
        if not self._is_fitted:
            return {}
        return dict(self._vectorizer.vocabulary_)

    @property
    def feature_names(self) -> list[str]:
        """Return ordered list of n-gram feature names."""
        if not self._is_fitted:
            return []
        return list(self._vectorizer.get_feature_names_out())

    @property
    def n_features(self) -> int:
        """Number of features in the fitted vocabulary."""
        return len(self.vocabulary)

    def top_ngrams(self, n: int = 20) -> list[tuple[str, float]]:
        """Return top-*n* n-grams by IDF weight."""
        if not self._is_fitted:
            return []
        names = self._vectorizer.get_feature_names_out()
        idfs = self._vectorizer.idf_
        order = np.argsort(idfs)[::-1][:n]
        return [(names[i], float(idfs[i])) for i in order]
