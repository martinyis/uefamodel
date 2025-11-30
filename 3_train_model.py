"""
Step 3: Train Multinomial Logistic Regression Model
Trains the model with proper train/test split and cross-validation
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
import pickle
from pathlib import Path

# Create directories
Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

def prepare_data(df, test_split_ratio=0.2):
    """
    Prepare training and test sets with temporal split
    """
    print("\nPreparing data...")
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Remove matches without sufficient history
    df = df[df['home_matches_played'] >= 3].reset_index(drop=True)
    
    print(f"Total matches with sufficient history: {len(df)}")
    
    # Temporal split: train on earlier matches, test on most recent
    split_idx = int(len(df) * (1 - test_split_ratio))
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Training set: {len(train_df)} matches ({train_df['Date'].min()} to {train_df['Date'].max()})")
    print(f"Test set: {len(test_df)} matches ({test_df['Date'].min()} to {test_df['Date'].max()})")
    
    # Feature columns (exclude metadata)
    feature_cols = [col for col in df.columns if col not in 
                   ['MatchID', 'Date', 'HomeTeam', 'AwayTeam', 'Result']]
    
    X_train = train_df[feature_cols]
    y_train = train_df['Result']
    
    X_test = test_df[feature_cols]
    y_test = test_df['Result']
    
    print(f"\nFeatures used: {len(feature_cols)}")
    print("Sample features:", feature_cols[:5])
    
    # Check for any NaN or infinite values
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    return X_train, X_test, y_train, y_test, feature_cols, train_df, test_df

def train_baseline_model(y_train, y_test):
    """
    Calculate baseline accuracy (always predict most common class)
    """
    most_common = y_train.mode()[0]
    baseline_acc = (y_test == most_common).mean()
    
    print("\n" + "="*60)
    print("BASELINE MODEL (Always predict most common outcome)")
    print("="*60)
    print(f"Most common outcome in training: {most_common}")
    print(f"Baseline accuracy on test set: {baseline_acc:.4f}")
    
    return baseline_acc

def train_logistic_model(X_train, y_train, X_test, y_test):
    """
    Train multinomial logistic regression with hyperparameter tuning
    """
    print("\n" + "="*60)
    print("TRAINING MULTINOMIAL LOGISTIC REGRESSION")
    print("="*60)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter grid for L2 regularization (Ridge)
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],  # Inverse of regularization strength
    }
    
    # Use TimeSeriesSplit for cross-validation (respects temporal ordering)
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("\nPerforming cross-validation to find best regularization strength...")
    
    # Multinomial logistic regression
    base_model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\nBest regularization parameter (C): {grid_search.best_params_['C']}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    # Best model
    best_model = grid_search.best_estimator_
    
    # Predictions
    y_train_pred = best_model.predict(X_train_scaled)
    y_test_pred = best_model.predict(X_test_scaled)
    
    # Probability predictions
    y_train_proba = best_model.predict_proba(X_train_scaled)
    y_test_proba = best_model.predict_proba(X_test_scaled)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    train_logloss = log_loss(y_train, y_train_proba)
    test_logloss = log_loss(y_test, y_test_proba)
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE")
    print("="*60)
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"\nTraining Log Loss: {train_logloss:.4f}")
    print(f"Test Log Loss: {test_logloss:.4f}")
    
    print("\n" + "="*60)
    print("DETAILED TEST SET RESULTS")
    print("="*60)
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, 
                                target_names=['Away Win', 'Draw', 'Home Win'] if set(y_test) == {'A', 'D', 'H'} else None))
    
    print("\nConfusion Matrix:")
    print("Rows = Actual, Columns = Predicted")
    cm = confusion_matrix(y_test, y_test_pred)
    classes = sorted(y_test.unique())
    print(f"Classes: {classes}")
    print(cm)
    
    # Save model and scaler
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'feature_names': X_train.columns.tolist(),
        'classes': best_model.classes_.tolist(),
        'test_accuracy': test_acc,
        'test_logloss': test_logloss
    }
    
    with open('models/logistic_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("\n✓ Model saved to models/logistic_model.pkl")
    
    return best_model, scaler, y_test_pred, y_test_proba

def analyze_feature_importance(model, feature_names, top_n=15):
    """
    Analyze and display feature importance from logistic regression coefficients
    """
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Get coefficients for each class
    classes = model.classes_
    coef = model.coef_
    
    print(f"\nClasses: {classes}")
    
    for i, class_name in enumerate(classes):
        print(f"\n{'-'*60}")
        print(f"Top {top_n} features predicting: {class_name}")
        print(f"{'-'*60}")
        
        # Get coefficients for this class
        class_coef = coef[i]
        
        # Sort by absolute value
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': class_coef,
            'Abs_Coefficient': np.abs(class_coef)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print(feature_importance.head(top_n).to_string(index=False))
    
    # Save feature importance
    feature_importance_all = pd.DataFrame({
        'Feature': feature_names
    })
    
    for i, class_name in enumerate(classes):
        feature_importance_all[f'Coef_{class_name}'] = coef[i]
    
    feature_importance_all.to_csv('results/feature_importance.csv', index=False)
    print("\n✓ Feature importance saved to results/feature_importance.csv")

def main():
    """
    Main training pipeline
    """
    print("="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    # Load features
    input_file = "data/processed/features.csv"
    
    try:
        df = pd.read_csv(input_file)
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"Loaded {len(df)} matches from {input_file}")
    except FileNotFoundError:
        print(f"ERROR: {input_file} not found!")
        print("Run: python 2_feature_engineering.py first")
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_cols, train_df, test_df = prepare_data(df)
    
    # Baseline model
    baseline_acc = train_baseline_model(y_train, y_test)
    
    # Train logistic regression
    model, scaler, y_test_pred, y_test_proba = train_logistic_model(
        X_train, y_train, X_test, y_test
    )
    
    # Analyze feature importance
    analyze_feature_importance(model, feature_cols)
    
    # Save predictions for evaluation
    test_results = test_df[['MatchID', 'Date', 'HomeTeam', 'AwayTeam', 'Result']].copy()
    test_results['Predicted'] = y_test_pred
    
    # Add probabilities
    for i, class_name in enumerate(model.classes_):
        test_results[f'Prob_{class_name}'] = y_test_proba[:, i]
    
    test_results.to_csv('results/test_predictions.csv', index=False)
    print("\n✓ Test predictions saved to results/test_predictions.csv")
    
    print("\n" + "="*60)
    print("SUCCESS!")
    print("="*60)
    print("Next step: Run python 4_evaluate_model.py for detailed analysis")

if __name__ == "__main__":
    main()


