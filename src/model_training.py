import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

def model_training(df):
    # Príprava dát
    X = df.drop(columns=['price', 'address'])
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with mlflow.start_run():
        model_gb = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=4,
                                            subsample=0.8, min_samples_split=2, random_state=42)

        mlflow.log_param("model_type", "GradientBoostingRegressor")
        mlflow.log_param("n_estimators", 150)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("max_depth", 4)
        mlflow.log_param("subsample", 0.8)
        mlflow.log_param("min_samples_split", 2)

        cv_scores = cross_val_score(model_gb, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
        avg_cv_mae = -cv_scores.mean()
        print(f"Priemerný MAE (cross-validation): {avg_cv_mae}")

        mlflow.log_metric("avg_cv_mae", avg_cv_mae)

        model_gb.fit(X_train_scaled, y_train)

        y_pred_gb = model_gb.predict(X_test_scaled)
        test_mae_gb = mean_absolute_error(y_test, y_pred_gb)
        test_rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
        test_r2_gb = r2_score(y_test, y_pred_gb)

        mlflow.log_metric("test_mae", test_mae_gb)
        mlflow.log_metric("test_rmse", test_rmse_gb)
        mlflow.log_metric("test_r2", test_r2_gb)

        mlflow.sklearn.log_model(model_gb, "gradient_boosting_model")

        print(f"Testovací MAE (GB): {test_mae_gb}")
        print(f"Testovací RMSE (GB): {test_rmse_gb}")
        print(f"Testovací R² (GB): {test_r2_gb * 100:.2f}%")

    return model_gb