"""End-to-end ML pipeline for telco churn classification.

Steps:
1. Load and preprocess data (missing values, encoding, scaling).
2. Split data 70/30 (random_state=42).
3. Train and tune several classifiers with GridSearchCV.
4. Evaluate on test set and compare metrics.
5. Save results and best model.

Designed for quick runs; grids are small to keep runtime reasonable.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

DATA_PATH = Path('telco_data.csv')
OUT_DIR = Path('ml_pipeline_output')
OUT_DIR.mkdir(exist_ok=True)


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df


def preprocess_split(df: pd.DataFrame):
    # Target
    if 'Churn' not in df.columns:
        raise RuntimeError('Churn column not found')
    df = df.copy()
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).astype(int)

    # Identify numeric and categorical
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'Churn']
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Preprocessing pipelines
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    # Create OneHotEncoder with compatibility across scikit-learn versions
    try:
        onehot = OneHotEncoder(handle_unknown='ignore', sparse=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', onehot),
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipe, numeric_cols),
        ('cat', categorical_pipe, categorical_cols),
    ])

    X = df.drop(columns=['Churn'])
    y = df['Churn'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    return preprocessor, X_train, X_test, y_train, y_test


def build_model_grid():
    models = {
        'LogisticRegression': (
            LogisticRegression(solver='liblinear', max_iter=1000),
            {'clf__C': [0.01, 0.1, 1.0]},
        ),
        'DecisionTree': (
            DecisionTreeClassifier(random_state=42),
            {'clf__max_depth': [3, 5, 10], 'clf__min_samples_leaf': [1, 5]},
        ),
        'RandomForest': (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {'clf__n_estimators': [100], 'clf__max_depth': [6, 10]},
        ),
        'SVC': (
            SVC(probability=True, random_state=42),
            {'clf__C': [0.1, 1.0], 'clf__kernel': ['rbf']},
        ),
        'KNN': (
            KNeighborsClassifier(),
            {'clf__n_neighbors': [3, 5, 7]},
        ),
        'GradientBoosting': (
            GradientBoostingClassifier(random_state=42),
            {'clf__n_estimators': [100], 'clf__learning_rate': [0.1, 0.05]},
        ),
    }
    return models


def evaluate_model(model, X_test_proc, y_test):
    preds = model.predict(X_test_proc)
    probs = None
    if hasattr(model, 'predict_proba'):
        try:
            probs = model.predict_proba(X_test_proc)[:, 1]
        except Exception:
            probs = None
    return {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall': recall_score(y_test, preds, zero_division=0),
        'f1_score': f1_score(y_test, preds, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, preds).tolist(),
        'preds': preds,
        'probs': probs,
    }


def main():
    df = load_data()
    preprocessor, X_train, X_test, y_train, y_test = preprocess_split(df)
    print('Data shapes — train:', X_train.shape, 'test:', X_test.shape)

    models = build_model_grid()
    results = []
    best_model = None
    best_f1 = -1.0
    best_name = None

    for name, (estimator, grid) in models.items():
        print('\nTraining', name)
        pipe = Pipeline([
            ('pre', preprocessor),
            ('clf', estimator)
        ])
        gs = GridSearchCV(pipe, param_grid=grid, cv=3, scoring='f1', n_jobs=-1)
        gs.fit(X_train, y_train)
        print(' Best params:', gs.best_params_)
        # Evaluate on test set
        eval_res = evaluate_model(gs.best_estimator_, X_test, y_test)
        eval_res.update({'model': name, 'best_params': gs.best_params_})
        results.append(eval_res)

        if eval_res['f1_score'] > best_f1:
            best_f1 = eval_res['f1_score']
            best_model = gs.best_estimator_
            best_name = name

    # Create comparison table
    rows = []
    for r in results:
        rows.append({
            'model': r['model'],
            'accuracy': r['accuracy'],
            'precision': r['precision'],
            'recall': r['recall'],
            'f1_score': r['f1_score'],
            'confusion_matrix': r['confusion_matrix'],
            'best_params': r['best_params'],
        })
    df_res = pd.DataFrame(rows).sort_values('f1_score', ascending=False)
    print('\n=== Model comparison ===')
    print(df_res[['model', 'accuracy', 'precision', 'recall', 'f1_score']])
    df_res.to_csv(OUT_DIR / 'model_comparison.csv', index=False)

    # Save best model
    if best_model is not None:
        joblib.dump(best_model, OUT_DIR / 'best_model.pkl')
        print('\nBest model:', best_name, 'with F1 =', best_f1)
        print('Saved best model to', OUT_DIR / 'best_model.pkl')

    # Save detailed results
    pd.DataFrame(results).to_json(OUT_DIR / 'detailed_results.json', orient='records')


if __name__ == '__main__':
    main()
