import pytest
import pandas as pd
import os
from src.load_data import load_data

def test_load_data_with_real_file():
    """Test načítania CSV súboru."""
    file_path = "./data/test_data.csv"
    df = load_data(file_path)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ["col1", "col2"]