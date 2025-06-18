import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os
from pathlib import Path

# 1. DATA LOADING ==============================================================

def load_data(filepath):
    """Load data with comprehensive error handling"""
    try:
        # Convert to Path object and verify existence
        data_path = Path(filepath)
        if not data_path.exists():
            available_files = "\n".join(sorted(data_path.parent.glob("*")))
            raise FileNotFoundError(
                f"Data file not found at: {data_path}\n"
                f"Available files in directory:\n{available_files}"
            )
        
        print(f"Loading data from: {data_path}")
        
        # Try reading the file
        df = pd.read_csv(data_path)
        print(f"Success! Loaded {len(df)} records with {len(df.columns)} columns")
        
        return df
        
    except Exception as e:
        print("\n" + "="*50)
        print("ERROR LOADING DATA:", str(e))
        print("="*50)
        print("\nTROUBLESHOOTING GUIDE:")
        print(f"1. Confirm the file exists at: {data_path}")
        print(f"2. Check file permissions (try opening it manually)")
        print(f"3. Verify file is CSV format (not Excel or other format)")
        print(f"4. Current working directory: {os.getcwd()}")
        if data_path.parent.exists():
            print(f"\nFiles in data directory:\n{os.listdir(data_path.parent)}")
        raise

# 2. DATA PREPROCESSING =======================================================

def preprocess_data(df):
    """Main preprocessing pipeline"""
    print("\n" + "="*50)
    print("STARTING DATA PREPROCESSING")
    print("="*50)
    
    # Display available columns
    print("\nAvailable columns:")
    print(df.columns.tolist())
    
    # Define features to use - MODIFY THESE BASED ON YOUR ACTUAL COLUMNS
    features = [
        'Province', 'PostalCode', 'Gender', 'VehicleType', 'Make',
        'RegistrationYear', 'SumInsured', 'ExcessSelected', 'CoverType'
    ]
    
    # Filter to only keep columns that exist
    features = [col for col in features if col in df.columns]
    print("\nUsing features:", features)
    
    # Filter only policies with claims
    if 'TotalClaims' not in df.columns:
        raise KeyError("'TotalClaims' column not found - required for analysis")
    
    claim_data = df[df['TotalClaims'] > 0].copy()
    print(f"\nFound {len(claim_data)} records with claims (original: {len(df)})")
    
    if len(claim_data) == 0:
        raise ValueError("No claims data available for modeling")
    
    # Split data
    X = claim_data[features]
    y = claim_data['TotalClaims']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print("\nData split complete:")
    print(f"- Training set: {len(X_train)} records")
    print(f"- Test set: {len(X_test)} records")
    
    # Define numeric and categorical features
    numeric_features = [col for col in features if pd.api.types.is_numeric_dtype(X[col])]
    categorical_features = list(set(features) - set(numeric_features))
    
    print("\nFeature types identified:")
    print(f"- Numeric: {numeric_features}")
    print(f"- Categorical: {categorical_features}")
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    print("\nPreprocessor configured successfully!")
    return preprocessor, X_train, X_test, y_train, y_test

# 3. SAVE OUTPUTS =============================================================

def save_outputs(preprocessor, X_train, X_test, y_train, y_test):
    """Save all preprocessing artifacts"""
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*50)
    print("SAVING OUTPUT FILES")
    print("="*50)
    
    # Save artifacts
    artifacts = {
        'preprocessor.joblib': preprocessor,
        'X_train.csv': X_train,
        'X_test.csv': X_test,
        'y_train.csv': y_train,
        'y_test.csv': y_test
    }
    
    for filename, obj in artifacts.items():
        filepath = output_dir / filename
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            obj.to_csv(filepath, index=False)
        else:
            joblib.dump(obj, filepath)
        print(f"- Saved {filename}")
    
    print("\nAll outputs saved to 'output' directory")

# MAIN EXECUTION ==============================================================

if __name__ == "__main__":
    try:
        # 1. Configure paths - UPDATE THIS TO YOUR ACTUAL PATH
        filepath = "D:/Project/ACIS_Insurance_Analytics/data/clean.csv"
        
        # 2. Load data
        df = load_data(filepath)
        
        # 3. Preprocess data
        preprocessor, X_train, X_test, y_train, y_test = preprocess_data(df)
        
        # 4. Save outputs
        save_outputs(preprocessor, X_train, X_test, y_train, y_test)
        
        print("\n" + "="*50)
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*50)
        
    except Exception as e:
        print("\n" + "="*50)
        print("PROCESS FAILED:", str(e))
        print("="*50)
        print("\nNext steps:")
        print("1. Check the error message above")
        print("2. Verify your data file format and contents")
        print("3. Ensure all required columns are present")
        print("4. Check file permissions and paths")