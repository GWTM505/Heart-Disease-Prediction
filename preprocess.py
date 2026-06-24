import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(filepath_or_url):
    """Loads the heart disease dataset from a file path or URL."""
    print(f"Loading data from: {filepath_or_url}")
    return pd.read_csv(filepath_or_url)

def get_data_summary(df):
    """Prints basic summary statistics and checks for missing values."""
    print("\n=== Dataset Summary ===")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Check for missing values
    missing = df.isnull().sum()
    print("\nMissing Values per Feature:")
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")
    
    # Check target distribution
    if 'target' in df.columns:
        target_counts = df['target'].value_counts()
        target_pct = df['target'].value_counts(normalize=True) * 100
        print("\nTarget Distribution:")
        for val, count in target_counts.items():
            print(f"  Class {val}: {count} ({target_pct[val]:.2f}%)")
            
    # Check duplicate rows
    duplicates_count = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicates_count} ({duplicates_count / len(df) * 100:.2f}%)")
    
    return {
        "shape": df.shape,
        "missing_values": missing.to_dict(),
        "duplicates": int(duplicates_count)
    }

def preprocess_pipeline(df, remove_duplicates=True, test_size=0.2, random_state=42):
    """
    Executes the entire preprocessing workflow:
    1. Removes duplicates (highly recommended to prevent data leakage in splits)
    2. Separates target and features
    3. Performs train-test split
    4. Identifies continuous vs categorical variables
    5. Scales continuous variables and one-hot encodes categorical variables
    """
    cleaned_df = df.copy()
    if remove_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
        print(f"\n[Preprocessing] Removed {initial_rows - len(cleaned_df)} duplicate rows. New shape: {cleaned_df.shape}")
    else:
        print("\n[Preprocessing] Keeping all duplicate rows (WARNING: potential data leakage between splits).")
        
    X = cleaned_df.drop(columns=['target'])
    y = cleaned_df['target']
    
    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[Preprocessing] Split data: Train shape = {X_train.shape}, Test shape = {X_test.shape}")
    
    # Identify feature types
    # Numerical features (continuous variables to be scaled)
    num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    # Multi-class categorical features (to be one-hot encoded)
    # cp (4 levels), restecg (3 levels), slope (3 levels), ca (5 levels), thal (4 levels)
    cat_features = ['cp', 'restecg', 'slope', 'ca', 'thal']
    
    # Binary/boolean features (keep as is since they are already 0/1)
    bin_features = ['sex', 'fbs', 'exang']
    
    # Column Transformer for scaling and encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)
        ],
        remainder='passthrough' # Leaves bin_features (sex, fbs, exang) untouched
    )
    
    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after one-hot encoding for feature importance analysis
    encoded_cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features)
    feature_names = num_features + list(encoded_cat_names) + bin_features
    
    # Convert back to DataFrame for easier inspection/manipulation
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)
    
    return X_train_df, X_test_df, y_train, y_test, feature_names
