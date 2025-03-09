import pandas as pd
from src.utils import extract_room_info

def preprocessing_data(df):
    df.drop(columns=['Unnamed: 0', 'id', 'name', 'hourse_rules'], inplace=True)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df['has_rating'] = df['rating'].notna().astype(int)
    avg_rating = df["rating"].mean()
    df["rating"] = df["rating"].fillna(avg_rating)
    df.drop(columns=['host_name', 'host_id'], axis=1, inplace=True)

    df[['guests', 'bedrooms', 'beds', 'bathrooms']] = df['features'].apply(lambda x: pd.Series(extract_room_info(x)))
    df.drop(columns=['features'], inplace=True)
    df.drop(columns=['amenities', 'safety_rules', 'img_links', 'checkin', 'checkout', 'country'], inplace=True)
    df['reviews'] = pd.to_numeric(df['reviews'], errors='coerce')
    df['reviews'] = df['reviews'].fillna(0).astype(int)
    df['z_score'] = (df['price'] - df['price'].mean()) / df['price'].std()
    outliers = df[df['z_score'].abs() > 3]
    df = df[df['z_score'].abs() <= 3]

    return df



