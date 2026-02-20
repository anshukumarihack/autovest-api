import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_autovest_model():
    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv("autovest_data.csv")
    
    X = df.drop(columns=['invest_today'])
    y = df['invest_today']
    
    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Define Preprocessing Steps
    numeric_features = ['daily_spending', 'spare_change_total', 'spending_variance', 
                        'emergency_balance_ratio', 'market_risk_score']
    categorical_features = ['user_type']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # 4. Build Pipeline
    # RandomForest is highly effective for tabular data and non-linear rule boundaries
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
    ])
    
    # 5. Train Model
    print("Training Random Forest pipeline...")
    pipeline.fit(X_train, y_train)
    
    # 6. Evaluate Model
    y_pred = pipeline.predict(X_test)
    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    
    # 7. Extract Feature Importance
    # We get the feature names after one-hot encoding
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    importances = pipeline.named_steps['classifier'].feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\n--- Feature Importance ---")
    print(feature_importance_df.to_string(index=False))
    
    # 8. Save Model
    model_path = "autovest_model.pkl"
    joblib.dump(pipeline, model_path)
    print(f"\nModel successfully saved to {model_path}")

if __name__ == "__main__":
    train_autovest_model()