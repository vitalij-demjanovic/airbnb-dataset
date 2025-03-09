from src.load_data import load_data
from src.model_training import model_training
from src.preprocessing import preprocessing_data


def main():
    path = './data/raw/airbnb-raw.csv'

    data = load_data(path)
    df = preprocessing_data(data)
    model = model_training(df)

    return model


if __name__ == "__main__":
    main()