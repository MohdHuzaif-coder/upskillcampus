"""
Prediction of Agriculture Crop Production in India
=====================================================
An end-to-end Machine Learning project that predicts crop production (in tonnes)
based on features like State, District, Crop, Season, Crop Year, and Area.

Dataset:
--------
This script is built around the popular "Crop Production in India" dataset
(commonly found on Kaggle: https://www.kaggle.com/datasets/abhinand05/crop-production-in-india)
Expected columns:
    State_Name, District_Name, Crop_Year, Season, Crop, Area, Production

If your CSV has different column names, just update COLUMN NAMES section below.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ======================================================================
# 1. LOAD DATA
# ======================================================================

DATA_PATH = "crop_production.csv"   # <-- change to your file path

def load_data(path=DATA_PATH):
    """Load the crop production dataset."""
    df = pd.read_csv(path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


# ======================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ======================================================================

def explore_data(df):
    print("\n--- Basic Info ---")
    print(df.info())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Statistical Summary ---")
    print(df.describe())

    print("\n--- Unique Crops ---")
    print(df['Crop'].nunique(), "unique crops")

    print("\n--- Unique States ---")
    print(df['State_Name'].nunique(), "unique states")

    # Top 10 crops by total production
    top_crops = df.groupby('Crop')['Production'].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_crops.values, y=top_crops.index, palette="viridis")
    plt.title("Top 10 Crops by Total Production in India")
    plt.xlabel("Total Production (tonnes)")
    plt.tight_layout()
    plt.savefig("top_10_crops.png")
    plt.close()

    # Top 10 states by total production
    top_states = df.groupby('State_Name')['Production'].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_states.values, y=top_states.index, palette="mako")
    plt.title("Top 10 States by Total Production")
    plt.xlabel("Total Production (tonnes)")
    plt.tight_layout()
    plt.savefig("top_10_states.png")
    plt.close()

    print("\nEDA plots saved as 'top_10_crops.png' and 'top_10_states.png'")


# ======================================================================
# 3. DATA CLEANING & PREPROCESSING
# ======================================================================

def clean_data(df):
    df = df.copy()

    # Drop rows with missing target (Production) or Area
    df = df.dropna(subset=['Production', 'Area'])

    # Remove rows where Production or Area is zero/negative (invalid entries)
    df = df[(df['Production'] > 0) & (df['Area'] > 0)]

    # Optional: remove extreme outliers using IQR on Production
    Q1 = df['Production'].quantile(0.25)
    Q3 = df['Production'].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df = df[(df['Production'] >= lower) & (df['Production'] <= upper)]

    print(f"Data shape after cleaning: {df.shape}")
    return df


def feature_engineering(df):
    df = df.copy()

    # Yield = Production / Area (helpful engineered feature, dropped before modeling
    # to avoid leakage since it's derived directly from target)
    df['Yield'] = df['Production'] / df['Area']

    # Encode categorical columns
    cat_cols = ['State_Name', 'District_Name', 'Season', 'Crop']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str).str.strip()
        df[col + '_enc'] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


# ======================================================================
# 4. MODEL BUILDING
# ======================================================================

FEATURES = ['State_Name_enc', 'District_Name_enc', 'Crop_Year',
            'Season_enc', 'Crop_enc', 'Area']
TARGET = 'Production'


def prepare_model_data(df):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


def evaluate_model(name, y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\n{name} Performance:")
    print(f"  R2 Score : {r2:.4f}")
    print(f"  MAE      : {mae:,.2f}")
    print(f"  RMSE     : {rmse:,.2f}")
    return {"model": name, "r2": r2, "mae": mae, "rmse": rmse}


def train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    results = []
    trained_models = {}

    # 1. Linear Regression (uses scaled features)
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    pred = lr.predict(X_test_scaled)
    results.append(evaluate_model("Linear Regression", y_test, pred))
    trained_models['Linear Regression'] = lr

    # 2. Decision Tree
    dt = DecisionTreeRegressor(max_depth=12, random_state=42)
    dt.fit(X_train, y_train)
    pred = dt.predict(X_test)
    results.append(evaluate_model("Decision Tree", y_test, pred))
    trained_models['Decision Tree'] = dt

    # 3. Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    results.append(evaluate_model("Random Forest", y_test, pred))
    trained_models['Random Forest'] = rf

    # 4. Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    pred = gb.predict(X_test)
    results.append(evaluate_model("Gradient Boosting", y_test, pred))
    trained_models['Gradient Boosting'] = gb

    results_df = pd.DataFrame(results).sort_values(by="r2", ascending=False)
    print("\n=== Model Comparison ===")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]['model']
    best_model = trained_models[best_model_name]
    print(f"\nBest Model: {best_model_name}")

    return trained_models, results_df, best_model, best_model_name


def tune_random_forest(X_train, y_train):
    """Optional: Hyperparameter tuning for Random Forest using GridSearchCV."""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)
    print("Best Params:", grid.best_params_)
    print("Best CV R2 :", grid.best_score_)
    return grid.best_estimator_


# ======================================================================
# 5. FEATURE IMPORTANCE
# ======================================================================

def plot_feature_importance(model, feature_names, model_name="Model"):
    if not hasattr(model, "feature_importances_"):
        print(f"{model_name} does not support feature importances.")
        return

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances[idx], y=np.array(feature_names)[idx], palette="crest")
    plt.title(f"Feature Importance - {model_name}")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()
    print("Feature importance plot saved as 'feature_importance.png'")


# ======================================================================
# 6. PREDICTION FUNCTION FOR NEW INPUT
# ======================================================================

def predict_production(model, encoders, state, district, crop_year, season, crop, area):
    """
    Predict crop production for a new/unseen input.
    Note: unseen categories fall back to -1 (handled gracefully).
    """
    def safe_encode(le, value):
        value = str(value).strip()
        if value in le.classes_:
            return le.transform([value])[0]
        else:
            print(f"Warning: '{value}' not seen during training. Using fallback encoding.")
            return -1

    input_data = pd.DataFrame([{
        'State_Name_enc': safe_encode(encoders['State_Name'], state),
        'District_Name_enc': safe_encode(encoders['District_Name'], district),
        'Crop_Year': crop_year,
        'Season_enc': safe_encode(encoders['Season'], season),
        'Crop_enc': safe_encode(encoders['Crop'], crop),
        'Area': area
    }])

    prediction = model.predict(input_data)[0]
    return prediction


# ======================================================================
# 7. MAIN PIPELINE
# ======================================================================

def main():
    # Step 1: Load
    df = load_data()

    # Step 2: EDA
    explore_data(df)

    # Step 3: Clean
    df = clean_data(df)

    # Step 4: Feature Engineering
    df, encoders = feature_engineering(df)

    # Step 5: Prepare train/test data
    X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler = prepare_model_data(df)

    # Step 6: Train models & compare
    trained_models, results_df, best_model, best_model_name = train_models(
        X_train, X_test, y_train, y_test, X_train_s, X_test_s
    )

    # Step 7: Feature importance (for tree-based best model)
    plot_feature_importance(best_model, FEATURES, best_model_name)

    # Step 8: Save best model & encoders
    joblib.dump(best_model, "best_crop_production_model.pkl")
    joblib.dump(encoders, "label_encoders.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print(f"\nSaved best model ({best_model_name}) and encoders to disk.")

    # Step 9: Example prediction
    sample_prediction = predict_production(
        best_model, encoders,
        state="Punjab", district="AMRITSAR", crop_year=2013,
        season="Kharif     ", crop="Rice", area=5000
    )
    print(f"\nExample Prediction -> Predicted Production: {sample_prediction:,.2f} tonnes")


if __name__ == "__main__":
    main()
