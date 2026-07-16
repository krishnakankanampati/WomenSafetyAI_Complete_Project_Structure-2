"""
====================================================
Women Safety AI Project
Module : Text Augmentation (EDA-lite)
File   : text_augment.py

Corpus-independent text perturbations (random deletion, random
swap, random insertion) used by balance_dataset.py to diversify
oversampled duplicate rows for minority classes, instead of
inserting exact duplicate text into training data. No external
corpora (e.g. WordNet) required, since offline SSL blocks
downloading them in this environment.
====================================================
"""

import random


def _split(text):
    return text.split()


def random_deletion(words, p=0.1):
    if len(words) <= 3:
        return words
    kept = [w for w in words if random.random() > p]
    return kept if kept else words


def random_swap(words, n=1):
    words = words.copy()
    length = len(words)
    if length < 2:
        return words
    for _ in range(n):
        i, j = random.sample(range(length), 2)
        words[i], words[j] = words[j], words[i]
    return words


def random_insertion(words, n=1):
    words = words.copy()
    length = len(words)
    if length < 1:
        return words
    for _ in range(n):
        pos = random.randint(0, len(words))
        words.insert(pos, random.choice(words))
    return words


def augment_text(text, seed=None):
    """Apply a random combination of deletion/swap/insertion to `text`."""
    if seed is not None:
        random.seed(seed)

    words = _split(text)
    if len(words) < 4:
        return text

    ops = random.sample(
        [
            lambda w: random_deletion(w, p=0.12),
            lambda w: random_swap(w, n=max(1, len(w) // 10)),
            lambda w: random_insertion(w, n=1),
        ],
        k=random.randint(1, 2),
    )

    for op in ops:
        words = op(words)

    return " ".join(words)
