import logging

from sklearn.preprocessing import StandardScaler

from src.load_data import load_data
from src.preprocessing import preprocessing_data
from src.model_training import model_training
import mlflow

# Nastavenie logovania
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

    try:
        current_run_id = mlflow.active_run().info.run_id
        logging.info(f"Aktuálny run_id: {current_run_id}")
    except Exception as e:
        logging.error(f"Chyba pri získavaní run_id: {e}")
        return

    model_uri = f"runs:/{current_run_id}/gradient_boosting_model"
    logging.info(f"Model URI: {model_uri}")

    try:
        model_loaded = mlflow.sklearn.load_model(model_uri)
        logging.info("Model bol úspešne načítaný z MLflow.")
    except Exception as e:
        logging.error(f"Chyba pri načítaní modelu z MLflow: {e}")
        return

    try:
        X_test = df.drop(columns=['price', 'address'])
        X_test_scaled = StandardScaler().fit_transform(X_test)
        y_pred = model_loaded.predict(X_test_scaled)
        logging.info(f"Predikcie: {y_pred}")
    except Exception as e:
        logging.error(f"Chyba pri predikcii s načítaným modelom: {e}")
        return

    print("Model bol úspešne natrénovaný a uložený do MLflow. Predikcie boli vykonané.")

    return model


if __name__ == "__main__":
    model = main()