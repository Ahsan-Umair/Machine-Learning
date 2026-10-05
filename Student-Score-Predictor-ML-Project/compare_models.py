# ==========================================
# Import Libraries
# ==========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("student_performance.csv")

# ==========================================
# Features and Target
# ==========================================

X = df[
    [
        "weekly_self_study_hours",
        "attendance_percentage",
        "class_participation"
    ]
]

y = df["total_score"]

# ==========================================
# Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# ==========================================
# Evaluation Function
# ==========================================

def evaluate_model(name, model):
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("=" * 40)
    print(name)
    print("=" * 40)

    print(f"MAE  : {mae:.2f}")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.3f}")

    print()

# ==========================================
# Compare Models
# ==========================================

evaluate_model("Linear Regression", LinearRegression())

max_depth_values = list(range(1, 11))

for depth in max_depth_values:
    evaluate_model(
        f"Decision Tree Regressor (max_depth={depth})",
        DecisionTreeRegressor(random_state=42, max_depth=depth)
    )

evaluate_model(
    "KNN Regressor",
    make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=5)
    )
)

evaluate_model(
    "SVR",
    make_pipeline(
        StandardScaler(),
        SVR(kernel="rbf", C=10.0, epsilon=0.2)
    )
)

evaluate_model(
    "Random Forest Regressor",
    RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=12
    )
)
