import logging
from src.load_data import load_data
from src.preprocessing import preprocessing_data
from src.model_training import model_training

logging.basicConfig(
    filename='ml_project.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    path = './data/raw/airbnb-raw.csv'

    try:
        data = load_data(path)
        logging.info("Dáta boli načítané.")
    except Exception as e:
        logging.error(f"Chyba pri načítaní dát: {e}")
        return

    try:
        df = preprocessing_data(data)
        logging.info("Dáta boli spracované.")
    except Exception as e:
        logging.error(f"Chyba pri spracovaní dát: {e}")
        return

    try:
        model = model_training(df)
        logging.info("Model bol úspešne natrénovaný.")
    except Exception as e:
        logging.error(f"Chyba pri trénovaní modelu: {e}")
        return

    print("Model bol úspešne natrénovaný a predikcie boli vykonané.")

    return model


if __name__ == "__main__":
    model = main()