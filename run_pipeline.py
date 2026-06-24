import os
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocess import load_data, get_data_summary, preprocess_pipeline
from src.train import train_and_tune_models
from src.evaluate import evaluate_models, plot_feature_importance

def download_dataset(url, dest_path):
    """Downloads the dataset if not already present."""
    if not os.path.exists(dest_path):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        print(f"Downloading dataset from {url} to {dest_path}...")
        urllib.request.urlretrieve(url, dest_path)
        print("Download complete!")
    else:
        print(f"Dataset already exists at {dest_path}")

def run_eda(df, output_dir="outputs"):
    """Runs basic Exploratory Data Analysis and saves key figures."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    print("\n--- Running Exploratory Data Analysis ---")
    
    # 1. Target Distribution
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(x='target', data=df, palette='Set2')
    plt.title('Target Variable Distribution (Heart Disease vs No Disease)', fontsize=14, weight='bold', pad=15)
    plt.xlabel('Condition (0 = No Heart Disease, 1 = Heart Disease)', fontsize=12, labelpad=10)
    plt.ylabel('Count', fontsize=12, labelpad=10)
    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=11, weight='bold')
    plt.tight_layout()
    target_dist_path = os.path.join(output_dir, "target_distribution.png")
    plt.savefig(target_dist_path, dpi=300)
    plt.close()
    print(f"Saved target distribution plot to {target_dist_path}")
    
    # 2. Correlation Matrix
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    # Mask the upper triangle for cleaner look
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", cbar=True,
                square=True, linewidths=.5, cbar_kws={"shrink": .8},
                annot_kws={"size": 9})
    plt.title('Correlation Matrix of Heart Disease Features', fontsize=14, weight='bold', pad=15)
    plt.tight_layout()
    corr_path = os.path.join(output_dir, "correlation_matrix.png")
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"Saved correlation matrix heatmap to {corr_path}")
    
    # Print top correlated features with target
    target_corr = corr['target'].drop('target').sort_values(ascending=False)
    print("\nFeature Correlation with Target (Sorted):")
    for feat, val in target_corr.items():
        print(f"  {feat:10s}: {val:.4f}")
        
    return target_corr

def main():
    # Set paths
    dataset_url = "https://raw.githubusercontent.com/dileep-lingamallu/Heart-Disease-Prediction-Dataset/main/heart.csv"
    data_path = os.path.join("data", "heart.csv")
    output_dir = "outputs"
    
    # Download dataset
    download_dataset(dataset_url, data_path)
    
    # Load dataset
    df = load_data(data_path)
    
    # Dataset statistics
    get_data_summary(df)
    
    # Run EDA
    run_eda(df, output_dir)
    
    # Preprocess and split: Using Deduplication as recommended
    # We will evaluate on Deduplicated data to prevent data leakage in our splits.
    print("\n--- Preprocessing & Train-Test Split (Deduplicated Data) ---")
    X_train, X_test, y_train, y_test, feature_names = preprocess_pipeline(
        df, remove_duplicates=True, test_size=0.2, random_state=42
    )
    
    # Train models
    print("\n--- Training and Tuning Models ---")
    models, best_params = train_and_tune_models(X_train, y_train)
    
    # Evaluate models
    print("\n--- Evaluating Models on Test Set ---")
    comparison_df = evaluate_models(models, X_test, y_test, output_dir)
    
    # Plot feature importances
    print("\n--- Visualizing Feature Importance ---")
    plot_feature_importance(models, feature_names, output_dir)
    
    # Print Comparison Table
    print("\n=== Model Evaluation Comparison ===")
    print(comparison_df.to_string(index=False))
    
    # Find Best Model
    best_model_row = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
    best_model_name = best_model_row['Model']
    best_model_acc = best_model_row['Accuracy']
    print(f"\nBest Model identified by Accuracy: {best_model_name} ({best_model_acc*100:.2f}% accuracy)")
    
    # Run comparison with data duplicates kept (for verification and comparison)
    print("\n" + "="*50)
    print("RUNNING PIPELINE WITH DUPLICATED DATA (Original Kaggle Format)")
    print("WARNING: This has potential data leakage, but helps show what happens if duplicates are kept.")
    print("="*50)
    
    X_train_d, X_test_d, y_train_d, y_test_d, _ = preprocess_pipeline(
        df, remove_duplicates=False, test_size=0.2, random_state=42
    )
    
    models_d, _ = train_and_tune_models(X_train_d, y_train_d)
    comparison_df_d = evaluate_models(models_d, X_test_d, y_test_d, os.path.join(output_dir, "duplicated_data"))
    
    print("\n=== Model Evaluation Comparison (With Duplicated Data) ===")
    print(comparison_df_d.to_string(index=False))
    
    # Save comparison metrics to CSV for the notebook / markdown reports
    comparison_df.to_csv(os.path.join(output_dir, "model_comparison_deduplicated.csv"), index=False)
    comparison_df_d.to_csv(os.path.join(output_dir, "model_comparison_duplicated.csv"), index=False)
    print(f"\nSaved comparison CSVs to {output_dir}/")

if __name__ == "__main__":
    main()
