"""Ames CSV semantics shared by experiments and single-house prediction."""
import pandas as pd


INPUT_SCHEMA = "ames-v2-literal-none"


def read_ames_csv(path, **kwargs):
    """Keep the legitimate masonry category 'None'; NA and empty fields are missing."""
    return pd.read_csv(path, keep_default_na=False, na_values=["NA", ""], **kwargs)
