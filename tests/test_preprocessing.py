import pytest
import pandas as pd
from src.preprocessing import preprocessing_data

@pytest.fixture
def sample_data():
    data = {
        'Unnamed: 0': [1, 2, 3],
        'id': [101, 102, 103],
        'name': ['Apt 1', 'Apt 2', 'Apt 3'],
        'hourse_rules': ['No smoking', 'Pets allowed', 'Quiet hours'],
        'rating': [4.5, None, 3.8],
        'host_name': ['Alice', 'Bob', 'Charlie'],
        'host_id': [201, 202, 203],
        'features': ['2 guests · 1 bedroom · 1 bed · 1 bath',
                     '4 guests · 2 bedrooms · 3 beds · 2 baths',
                     '1 guest · 1 bedroom · 1 bed · 1 shared bath'],
        'amenities': ['Wifi, Kitchen', 'TV, Heater', 'Pool, Parking'],
        'safety_rules': ['Fire alarm', None, 'First aid kit'],
        'img_links': ['img1.jpg', 'img2.jpg', 'img3.jpg'],
        'checkin': ['14:00', '15:00', '13:00'],
        'checkout': ['11:00', '10:00', '12:00'],
        'country': ['USA', 'Canada', 'UK'],
        'reviews': [20, None, 5],
        'price': [100, 300, 150]
    }
    return pd.DataFrame(data)

def test_preprocessing(sample_data):
    """Test preprocessing_data function."""
    df_processed = preprocessing_data(sample_data)

    removed_columns = ['Unnamed: 0', 'id', 'name', 'hourse_rules',
                       'host_name', 'host_id', 'features', 'amenities',
                       'safety_rules', 'img_links', 'checkin', 'checkout', 'country']
    for col in removed_columns:
        assert col not in df_processed.columns, f"Column {col} should be removed."

    assert df_processed['rating'].isna().sum() == 0, "Rating should not have missing values."

    assert df_processed['reviews'].min() >= 0, "Reviews should be zero if missing."

    for col in ['guests', 'bedrooms', 'beds', 'bathrooms']:
        assert col in df_processed.columns, f"Column {col} is missing."

    assert df_processed['z_score'].abs().max() <= 3, "Outliers should be removed."