from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

def model_training(df):
    X = df.drop(columns=['price', 'address'])
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Škálovanie dát
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Definovanie modelu
    model_gb = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=4,
                                        subsample=0.8, min_samples_split=2, random_state=42)

    # Použitie cross-validation
    cv_scores = cross_val_score(model_gb, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')

    # Priemer MAE z cross-validation
    print(f"Priemerný MAE (cross-validation): {-cv_scores.mean()}")

    # Trénovanie modelu na celých trénovacích dátach
    model_gb.fit(X_train_scaled, y_train)

    # Testovanie modelu
    y_pred_gb = model_gb.predict(X_test_scaled)

    test_mae_gb = mean_absolute_error(y_test, y_pred_gb)
    test_rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
    test_r2_gb = r2_score(y_test, y_pred_gb)

    print(f"Testovací MAE (GB): {test_mae_gb}")
    print(f"Testovací RMSE (GB): {test_rmse_gb}")
    print(f"Testovací R² (GB): {test_r2_gb * 100:.2f}%")

