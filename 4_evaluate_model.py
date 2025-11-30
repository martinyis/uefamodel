"""
Step 4: Model Evaluation and Visualization
Creates calibration plots, performance visualizations, and detailed analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_model_and_results():
    """
    Load trained model and test predictions
    """
    # Load model
    with open('models/logistic_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    # Load predictions
    predictions = pd.read_csv('results/test_predictions.csv')
    predictions['Date'] = pd.to_datetime(predictions['Date'])
    
    return model_data, predictions

def plot_confusion_matrix(y_true, y_pred, classes, save_path='results/confusion_matrix.png'):
    """
    Create a visual confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Normalize to percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Actual Result', fontsize=12)
    ax1.set_xlabel('Predicted Result', fontsize=12)
    
    # Percentages
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax2)
    ax2.set_title('Confusion Matrix (Percentage)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Actual Result', fontsize=12)
    ax2.set_xlabel('Predicted Result', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")
    plt.close()

def plot_calibration_curve(y_true, y_proba, classes, save_path='results/calibration_curves.png'):
    """
    Create calibration curves for each outcome class
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (class_name, ax) in enumerate(zip(classes, axes)):
        # Binary indicator for this class
        y_binary = (y_true == class_name).astype(int)
        y_prob = y_proba[:, i]
        
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(y_binary, y_prob, n_bins=10, strategy='uniform')
        
        # Plot
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax.plot(prob_pred, prob_true, 'o-', linewidth=2, label=f'{class_name} Calibration')
        
        ax.set_xlabel('Predicted Probability', fontsize=12)
        ax.set_ylabel('Actual Frequency', fontsize=12)
        ax.set_title(f'Calibration: {class_name}', fontsize=13, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Calibration curves saved to {save_path}")
    plt.close()

def plot_probability_distribution(predictions, classes, save_path='results/probability_distribution.png'):
    """
    Plot distribution of predicted probabilities
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (class_name, ax) in enumerate(zip(classes, axes)):
        prob_col = f'Prob_{class_name}'
        
        # Separate by actual outcome
        for actual_class in classes:
            probs = predictions[predictions['Result'] == actual_class][prob_col]
            ax.hist(probs, bins=20, alpha=0.5, label=f'Actual: {actual_class}')
        
        ax.set_xlabel(f'Predicted Probability of {class_name}', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Probability Distribution: {class_name}', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Probability distribution saved to {save_path}")
    plt.close()

def plot_feature_importance(save_path='results/feature_importance.png'):
    """
    Visualize top feature importances
    """
    feature_imp = pd.read_csv('results/feature_importance.csv')
    
    # Get top features by absolute average coefficient
    coef_cols = [col for col in feature_imp.columns if col.startswith('Coef_')]
    feature_imp['Avg_Abs_Coef'] = feature_imp[coef_cols].abs().mean(axis=1)
    top_features = feature_imp.nlargest(15, 'Avg_Abs_Coef')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(top_features))
    width = 0.25
    
    colors = {'A': '#FF6B6B', 'D': '#FFD93D', 'H': '#6BCB77'}
    
    for i, col in enumerate(coef_cols):
        class_name = col.split('_')[1]
        values = top_features[col].values
        offset = (i - 1) * width
        ax.barh(x + offset, values, width, label=class_name, color=colors.get(class_name, 'gray'))
    
    ax.set_yticks(x)
    ax.set_yticklabels(top_features['Feature'].values)
    ax.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
    ax.legend(title='Outcome', loc='best')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Feature importance plot saved to {save_path}")
    plt.close()

def analyze_prediction_confidence(predictions, classes):
    """
    Analyze model confidence and accuracy relationship
    """
    print("\n" + "="*60)
    print("PREDICTION CONFIDENCE ANALYSIS")
    print("="*60)
    
    # Get maximum probability for each prediction
    prob_cols = [f'Prob_{c}' for c in classes]
    predictions['Max_Prob'] = predictions[prob_cols].max(axis=1)
    predictions['Correct'] = (predictions['Result'] == predictions['Predicted']).astype(int)
    
    # Bin by confidence level
    confidence_bins = [0, 0.4, 0.5, 0.6, 0.7, 1.0]
    bin_labels = ['<40%', '40-50%', '50-60%', '60-70%', '>70%']
    
    predictions['Confidence_Bin'] = pd.cut(predictions['Max_Prob'], 
                                           bins=confidence_bins, 
                                           labels=bin_labels)
    
    confidence_analysis = predictions.groupby('Confidence_Bin').agg({
        'Correct': ['sum', 'count', 'mean']
    }).round(3)
    
    confidence_analysis.columns = ['Correct', 'Total', 'Accuracy']
    
    print("\nAccuracy by Prediction Confidence:")
    print(confidence_analysis)
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    
    conf_data = confidence_analysis.reset_index()
    ax.bar(conf_data['Confidence_Bin'], conf_data['Accuracy'], color='steelblue', alpha=0.7)
    
    for i, (idx, row) in enumerate(conf_data.iterrows()):
        ax.text(i, row['Accuracy'] + 0.02, f"{row['Accuracy']:.2%}\n(n={int(row['Total'])})", 
                ha='center', fontsize=10)
    
    ax.set_xlabel('Prediction Confidence Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy vs. Prediction Confidence', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/confidence_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Confidence analysis saved to results/confidence_analysis.png")
    plt.close()
    
    return predictions

def analyze_by_outcome(predictions):
    """
    Analyze model performance for each outcome type
    """
    print("\n" + "="*60)
    print("PERFORMANCE BY OUTCOME TYPE")
    print("="*60)
    
    for outcome in predictions['Result'].unique():
        subset = predictions[predictions['Result'] == outcome]
        accuracy = (subset['Result'] == subset['Predicted']).mean()
        count = len(subset)
        
        print(f"\n{outcome} (n={count}):")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy:.1%})")
        print(f"  Predicted as:")
        print(subset['Predicted'].value_counts().to_string())

def generate_summary_report(model_data, predictions):
    """
    Generate a text summary report
    """
    classes = model_data['classes']
    
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("CHAMPIONS LEAGUE MATCH PREDICTION - MODEL EVALUATION SUMMARY")
    report_lines.append("="*70)
    report_lines.append("")
    
    # Overall metrics
    report_lines.append("OVERALL PERFORMANCE:")
    report_lines.append("-" * 70)
    report_lines.append(f"Test Accuracy: {model_data['test_accuracy']:.4f} ({model_data['test_accuracy']:.1%})")
    report_lines.append(f"Test Log Loss: {model_data['test_logloss']:.4f}")
    report_lines.append(f"Number of test matches: {len(predictions)}")
    report_lines.append("")
    
    # Baseline comparison
    most_common = predictions['Result'].mode()[0]
    baseline_acc = (predictions['Result'] == most_common).mean()
    improvement = (model_data['test_accuracy'] - baseline_acc) / baseline_acc * 100
    
    report_lines.append("BASELINE COMPARISON:")
    report_lines.append("-" * 70)
    report_lines.append(f"Baseline (always predict '{most_common}'): {baseline_acc:.4f}")
    report_lines.append(f"Our model: {model_data['test_accuracy']:.4f}")
    report_lines.append(f"Improvement: {improvement:.1f}%")
    report_lines.append("")
    
    # Per-class performance
    report_lines.append("PERFORMANCE BY OUTCOME:")
    report_lines.append("-" * 70)
    for outcome in sorted(predictions['Result'].unique()):
        subset = predictions[predictions['Result'] == outcome]
        accuracy = (subset['Result'] == subset['Predicted']).mean()
        count = len(subset)
        report_lines.append(f"{outcome}: {accuracy:.4f} (n={count})")
    report_lines.append("")
    
    # Save report
    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    
    with open('results/evaluation_summary.txt', 'w') as f:
        f.write(report_text)
    
    print("\n✓ Summary report saved to results/evaluation_summary.txt")

def main():
    """
    Main evaluation pipeline
    """
    print("="*60)
    print("MODEL EVALUATION & VISUALIZATION")
    print("="*60)
    
    # Load model and predictions
    try:
        model_data, predictions = load_model_and_results()
        print(f"✓ Loaded model and {len(predictions)} test predictions")
    except FileNotFoundError as e:
        print(f"ERROR: Required files not found!")
        print("Run: python 3_train_model.py first")
        return
    
    classes = model_data['classes']
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Confusion matrix
    plot_confusion_matrix(
        predictions['Result'], 
        predictions['Predicted'], 
        classes
    )
    
    # Calibration curves
    prob_cols = [f'Prob_{c}' for c in classes]
    y_proba = predictions[prob_cols].values
    plot_calibration_curve(
        predictions['Result'], 
        y_proba, 
        classes
    )
    
    # Probability distributions
    plot_probability_distribution(predictions, classes)
    
    # Feature importance
    plot_feature_importance()
    
    # Confidence analysis
    predictions = analyze_prediction_confidence(predictions, classes)
    
    # Outcome analysis
    analyze_by_outcome(predictions)
    
    # Summary report
    generate_summary_report(model_data, predictions)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print("Check the results/ folder for all plots and analysis")
    print("\nNext step: Run python 5_generate_results_section.py")

if __name__ == "__main__":
    main()


