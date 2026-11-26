"""Dataset loading and splitting.

- Bitext customer-support messages -> intent labels (classification)
- DialogSum dialogues -> human summaries (summarization)
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from ticketlens import config


def load_bitext() -> pd.DataFrame:
    """Download (or read from the HF cache) the Bitext dataset as a DataFrame.

    Returns columns: text, label, category.
    """
    from datasets import load_dataset  # imported lazily: heavy library

    ds = load_dataset(config.CLS_DATASET, split="train")
    df = ds.to_pandas()
    df = df.rename(columns={config.TEXT_COL: "text", config.LABEL_COL: "label"})
    df = df[["text", "label", "category"]].dropna()
    # Remove exact duplicate messages so the same sentence cannot be in train AND test.
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    return df


def stratified_split(df: pd.DataFrame, seed: int = config.SEED):
    """Split into train / val / test while keeping the label distribution identical.

    `stratify=` makes every split contain each intent in the same proportion,
    which matters for fair macro-F1 on small classes.
    """
    # First cut off the test set...
    train_val, test = train_test_split(df, test_size=config.TEST_FRAC, stratify=df["label"], random_state=seed)
    # ...then cut the validation set out of what is left.
    # VAL_FRAC is relative to the full data, so rescale it.
    val_size = config.VAL_FRAC / (1 - config.TEST_FRAC)
    train, val = train_test_split(train_val, test_size=val_size, stratify=train_val["label"], random_state=seed)
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def load_dialogsum_test(n: int = config.SUM_TEST_SIZE, seed: int = config.SEED) -> pd.DataFrame:
    """Return a fixed random subset of the DialogSum *test* split.

    Columns: id, dialogue, summary. The same seed always picks the same rows.
    """
    from datasets import load_dataset

    ds = load_dataset(config.SUM_DATASET, split="test")
    df = ds.to_pandas()[["id", "dialogue", "summary"]]
    if n < len(df):
        df = df.sample(n=n, random_state=seed)
    return df.reset_index(drop=True)
