"""Dataset loading utilities (stub).

Graph‑based NLP requires corpora annotated for various tasks such as
topic classification, sentiment analysis or hate speech detection.
Public datasets for Persian are scarce.  This module provides
convenience functions to load or simulate small datasets for testing
the Rakhshai Graph-based NLP library.  In practice, users should replace these
functions with loaders for real datasets.
"""

from __future__ import annotations

from typing import List, Tuple


def load_dummy_classification_dataset() -> Tuple[List[str], List[int]]:
    """Return a small dummy dataset for text classification.

    The dataset consists of a few Persian sentences and integer labels
    representing categories (e.g. 0=politics, 1=sports, 2=art).  This
    function is intended solely for testing the pipeline and does not
    reflect real linguistic phenomena.

    Returns
    -------
    Tuple[List[str], List[int]]
        A tuple ``(documents, labels)``.
    """
    docs = [
        "این یک خبر سیاسی است و دربارهٔ انتخابات صحبت می‌کند.",
        "تیم فوتبال امروز بازی مهمی دارد و همه منتظر نتیجه هستند.",
        "نمایشگاه جدید هنری با آثار نقاشان جوان افتتاح شد.",
    ]
    labels = [0, 1, 2]
    return docs, labels