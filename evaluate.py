import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve
)

def evaluate_models(models, X_test, y_test, output_dir="outputs"):
    """
    Evaluates all trained models on test data, compiles performance metrics,
    and generates visual plots (Confusion Matrices, ROC curves).
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics_list = []
    
    # We will plot ROC curves for all models on one plot
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    
    for name, model in models.items():
        # Predict class labels
        y_pred = model.predict(X_test)
        
        # Predict class probabilities (for ROC-AUC)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
            
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        metrics_list.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        })
        
        # Plot ROC curve for this model
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})", linewidth=2.5)
        
        # Save individual confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'],
                    annot_kws={"size": 14, "weight": "bold"})
        plt.title(f'Confusion Matrix - {name}', fontsize=14, weight='bold', pad=15)
        plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
        plt.ylabel('True Label', fontsize=12, labelpad=10)
        plt.tight_layout()
        cm_path = os.path.join(output_dir, f"confusion_matrix_{name.lower().replace(' ', '_')}.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"[Evaluation] Saved Confusion Matrix for {name} to {cm_path}")

    # Finalize and save the joint ROC Curve plot
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing", alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, labelpad=10)
    plt.ylabel('True Positive Rate', fontsize=12, labelpad=10)
    plt.title('ROC Curves Comparison', fontsize=14, weight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curves_comparison.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved ROC Curves Comparison to {roc_path}")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(metrics_list)
    return comparison_df

def plot_feature_importance(models, feature_names, output_dir="outputs"):
    """
    Plots feature importance for Random Forest (MDI) and Logistic Regression (Coefficients).
    Note: KNN does not have direct feature importance attributes.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Random Forest Feature Importance
    if 'Random Forest' in models:
        rf_model = models['Random Forest']
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette="viridis")
        plt.title('Feature Importances - Random Forest (MDI)', fontsize=14, weight='bold', pad=15)
        plt.xlabel('Mean Decrease in Impurity (MDI)', fontsize=12, labelpad=10)
        plt.ylabel('Features', fontsize=12, labelpad=10)
        plt.tight_layout()
        rf_imp_path = os.path.join(output_dir, "feature_importance_random_forest.png")
        plt.savefig(rf_imp_path, dpi=300)
        plt.close()
        print(f"[Evaluation] Saved Random Forest feature importances to {rf_imp_path}")
        
    # 2. Logistic Regression Coefficients
    if 'Logistic Regression' in models:
        lr_model = models['Logistic Regression']
        coefs = lr_model.coef_[0]
        abs_coefs = np.abs(coefs)
        indices = np.argsort(abs_coefs)[::-1]
        
        plt.figure(figsize=(12, 6))
        # Draw both actual coefficient direction (color coded) and magnitude
        colors = ['#4c72b0' if c > 0 else '#c44e52' for c in coefs[indices]]
        sns.barplot(x=coefs[indices], y=np.array(feature_names)[indices], palette=colors)
        plt.axvline(0, color='black', linestyle='--', alpha=0.7)
        plt.title('Model Coefficients (Feature Impact) - Logistic Regression', fontsize=14, weight='bold', pad=15)
        plt.xlabel('Coefficient Value (Positive: Promotes Heart Disease | Negative: Prevents)', fontsize=12, labelpad=10)
        plt.ylabel('Features', fontsize=12, labelpad=10)
        plt.tight_layout()
        lr_imp_path = os.path.join(output_dir, "feature_importance_logistic_regression.png")
        plt.savefig(lr_imp_path, dpi=300)
        plt.close()
        print(f"[Evaluation] Saved Logistic Regression coefficients to {lr_imp_path}")
