import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import shap
from pathlib import Path
import sklearn

# 1. LOAD PREPROCESSED DATA ==================================================

def load_preprocessed_data():
    """Load all artifacts from preprocessing"""
    try:
        # Load preprocessor
        preprocessor = joblib.load('output/preprocessor.joblib')
        
        # Load data splits
        X_train = pd.read_csv('output/X_train.csv')
        X_test = pd.read_csv('output/X_test.csv')
        y_train = pd.read_csv('output/y_train.csv').squeeze()
        y_test = pd.read_csv('output/y_test.csv').squeeze()
        
        print("Preprocessed data loaded successfully!")
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        return preprocessor, X_train, X_test, y_train, y_test
        
    except Exception as e:
        print("\nERROR LOADING PREPROCESSED DATA:", str(e))
        print("\nTROUBLESHOOTING:")
        print("1. Make sure you ran preprocess_task4.py first")
        print("2. Verify 'output' directory exists with these files:")
        print("   - preprocessor.joblib")
        print("   - X_train.csv, X_test.csv")
        print("   - y_train.csv, y_test.csv")
        print("3. Check paths are correct")
        raise

# 2. MODEL BUILDING ==========================================================

def calculate_rmse(y_true, y_pred):
    """Calculate RMSE compatible with all sklearn versions"""
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)

def build_models(preprocessor, X_train, y_train, X_test, y_test):
    """Build and evaluate multiple models"""
    results = {}
    
    # Linear Regression
    lr_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    lr_pipe.fit(X_train, y_train)
    y_pred = lr_pipe.predict(X_test)
    results['LinearRegression'] = {
        'RMSE': calculate_rmse(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
        'model': lr_pipe
    }
    
    # Random Forest
    rf_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    rf_pipe.fit(X_train, y_train)
    y_pred = rf_pipe.predict(X_test)
    results['RandomForest'] = {
        'RMSE': calculate_rmse(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
        'model': rf_pipe
    }
    
    # XGBoost
    xgb_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(objective='reg:squarederror', random_state=42))
    ])
    xgb_pipe.fit(X_train, y_train)
    y_pred = xgb_pipe.predict(X_test)
    results['XGBoost'] = {
        'RMSE': calculate_rmse(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
        'model': xgb_pipe
    }
    
    return results

# 3. MODEL EVALUATION ========================================================

def evaluate_models(results, y_test):
    """Evaluate and compare model performance"""
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Print metrics
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"- RMSE: {metrics['RMSE']:.2f}")
        print(f"- R2: {metrics['R2']:.2f}")
    
    # Plot feature importance for best model
    best_model_name = max(results.items(), key=lambda x: x[1]['R2'])[0]
    best_model = results[best_model_name]['model']
    
    print(f"\nBest model: {best_model_name}")
    
    # SHAP analysis for interpretability
    if hasattr(best_model.named_steps['regressor'], 'feature_importances_'):
        explain_model(best_model, X_test)

def explain_model(model, X_test):
    """Explain model using SHAP values"""
    print("\nGenerating SHAP explanations...")
    
    try:
        # Process test data through preprocessor
        preprocessor = model.named_steps['preprocessor']
        X_test_processed = preprocessor.transform(X_test)
        
        # Get feature names
        numeric_features = preprocessor.named_transformers_['num'].features
        categorical_features = preprocessor.named_transformers_['cat'].features
        
        # Get categorical feature names after one-hot encoding
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
        
        all_features = numeric_features + list(cat_feature_names)
        
        # SHAP analysis
        explainer = shap.Explainer(model.named_steps['regressor'])
        shap_values = explainer.shap_values(X_test_processed)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_processed, feature_names=all_features)
        plt.title(f'Feature Importance - SHAP Values')
        plt.tight_layout()
        plt.savefig('output/shap_summary.png')
        print("Saved SHAP summary plot to output/shap_summary.png")
    except Exception as e:
        print(f"\nCould not generate SHAP explanations: {str(e)}")

# MAIN EXECUTION =============================================================

if __name__ == "__main__":
    try:
        # Print version info for debugging
        print(f"scikit-learn version: {sklearn.__version__}")
        
        # 1. Load preprocessed data
        preprocessor, X_train, X_test, y_train, y_test = load_preprocessed_data()
        
        # 2. Build models
        print("\nBuilding models...")
        results = build_models(preprocessor, X_train, y_train, X_test, y_test)
        
        # 3. Evaluate models
        evaluate_models(results, y_test)
        
        # 4. Save best model
        best_model_name = max(results.items(), key=lambda x: x[1]['R2'])[0]
        joblib.dump(results[best_model_name]['model'], 'output/best_model.joblib')
        print(f"\nSaved best model ({best_model_name}) to output/best_model.joblib")
        
        print("\n" + "="*50)
        print("MODEL BUILDING COMPLETED SUCCESSFULLY!")
        print("="*50)
        
    except Exception as e:
        print("\n" + "="*50)
        print("MODEL BUILDING FAILED:", str(e))
        print("="*50)