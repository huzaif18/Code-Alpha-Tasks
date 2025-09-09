# ===============================================================================
# CODEALPHA INTERNSHIP - TASK 1: IRIS FLOWER CLASSIFICATION
# Author: Huzaif Ali Khan Bepari
# Date: AUGUST 2025
# Description: Professional ML classification model for Iris species prediction
# ===============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings

warnings.filterwarnings('ignore')

# Set professional styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🌸 CODEALPHA IRIS CLASSIFICATION PROJECT")
print("=" * 50)


# ===============================================================================
# 1. DATA LOADING AND EXPLORATION
# ===============================================================================

def load_iris_data(file_path):
    """Load and explore the Iris dataset"""

    print("📂 LOADING DATASET...")
    print(f"Dataset path: {file_path}")

    try:
        # Load the dataset
        df = pd.read_csv(file_path)

        print("\n📋 DATASET OVERVIEW")
        print("-" * 30)
        print(f"Dataset Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Missing Values: {df.isnull().sum().sum()}")

        # Display first few rows
        print("\n📊 SAMPLE DATA:")
        print(df.head())

        # Check species distribution
        print("\n🎯 SPECIES DISTRIBUTION:")
        print(df['Species'].value_counts())

        # Statistical summary
        print("\n📈 STATISTICAL SUMMARY:")
        print(df.describe())

        return df

    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found!")
        print("Please check the file path and try again.")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return None


# ===============================================================================
# 2. COMPREHENSIVE EDA
# ===============================================================================

def perform_iris_eda(df):
    """Comprehensive Exploratory Data Analysis for Iris data"""

    print("\n🔍 EXPLORATORY DATA ANALYSIS")
    print("-" * 40)

    # Create feature columns (excluding Id and Species)
    feature_cols = [col for col in df.columns if col not in ['Id', 'Species']]

    # Set up the plotting area
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Iris Dataset - Comprehensive EDA', fontsize=16, fontweight='bold')

    # 1. Species Distribution
    species_counts = df['Species'].value_counts()
    species_counts.plot(kind='bar', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Species Distribution')
    axes[0, 0].set_xlabel('Species')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. Correlation Heatmap
    correlation_matrix = df[feature_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[0, 1])
    axes[0, 1].set_title('Feature Correlation Matrix')

    # 3. Sepal Length vs Sepal Width scatter plot
    for species in df['Species'].unique():
        species_data = df[df['Species'] == species]
        axes[0, 2].scatter(species_data['SepalLengthCm'], species_data['SepalWidthCm'],
                           label=species.replace('Iris-', ''), alpha=0.7)
    axes[0, 2].set_xlabel('Sepal Length (cm)')
    axes[0, 2].set_ylabel('Sepal Width (cm)')
    axes[0, 2].set_title('Sepal Length vs Sepal Width')
    axes[0, 2].legend()

    # 4. Petal Length vs Petal Width scatter plot
    for species in df['Species'].unique():
        species_data = df[df['Species'] == species]
        axes[1, 0].scatter(species_data['PetalLengthCm'], species_data['PetalWidthCm'],
                           label=species.replace('Iris-', ''), alpha=0.7)
    axes[1, 0].set_xlabel('Petal Length (cm)')
    axes[1, 0].set_ylabel('Petal Width (cm)')
    axes[1, 0].set_title('Petal Length vs Petal Width')
    axes[1, 0].legend()

    # 5-8. Box plots for each feature
    feature_titles = ['Sepal Length Distribution', 'Sepal Width Distribution',
                      'Petal Length Distribution', 'Petal Width Distribution']

    for i, (feature, title) in enumerate(zip(feature_cols, feature_titles)):
        if i < 4:
            row = 1 + i // 2
            col = 1 + i % 2
            df.boxplot(column=feature, by='Species', ax=axes[row, col])
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel('Species')
            axes[row, col].set_ylabel(f'{feature} (cm)')

    # 9. Feature distributions
    df[feature_cols].hist(bins=20, figsize=(12, 8), ax=axes[2, 2])
    axes[2, 2].set_title('Feature Distributions')

    plt.tight_layout()
    plt.show()

    # Key insights
    print("\n💡 KEY EDA INSIGHTS:")
    print("✓ Perfect class balance (50 samples per species)")
    print("✓ Strong correlation between petal dimensions")
    print("✓ Setosa clearly separable from other species")
    print("✓ No missing values - clean dataset")
    print("✓ Petal features more discriminative than sepal features")


# ===============================================================================
# 3. FEATURE ENGINEERING
# ===============================================================================

def engineer_iris_features(df):
    """Create additional features for better model performance"""

    print("\n🔧 FEATURE ENGINEERING")
    print("-" * 30)

    df_enhanced = df.copy()

    # Create ratio features
    df_enhanced['sepal_ratio'] = df_enhanced['SepalLengthCm'] / df_enhanced['SepalWidthCm']
    df_enhanced['petal_ratio'] = df_enhanced['PetalLengthCm'] / df_enhanced['PetalWidthCm']

    # Create area features
    df_enhanced['sepal_area'] = df_enhanced['SepalLengthCm'] * df_enhanced['SepalWidthCm']
    df_enhanced['petal_area'] = df_enhanced['PetalLengthCm'] * df_enhanced['PetalWidthCm']

    # Create size comparison features
    df_enhanced['length_ratio'] = df_enhanced['PetalLengthCm'] / df_enhanced['SepalLengthCm']
    df_enhanced['width_ratio'] = df_enhanced['PetalWidthCm'] / df_enhanced['SepalWidthCm']

    # Create total size feature
    df_enhanced['total_area'] = df_enhanced['sepal_area'] + df_enhanced['petal_area']

    # Create perimeter approximations
    df_enhanced['sepal_perimeter'] = 2 * (df_enhanced['SepalLengthCm'] + df_enhanced['SepalWidthCm'])
    df_enhanced['petal_perimeter'] = 2 * (df_enhanced['PetalLengthCm'] + df_enhanced['PetalWidthCm'])

    print("✓ Created ratio features (sepal_ratio, petal_ratio)")
    print("✓ Created area features (sepal_area, petal_area)")
    print("✓ Created size comparison features (length_ratio, width_ratio)")
    print("✓ Created composite features (total_area, perimeters)")
    print(f"✓ Total features: {len([col for col in df_enhanced.columns if col not in ['Id', 'Species']])}")

    return df_enhanced


# ===============================================================================
# 4. MODEL DEVELOPMENT AND COMPARISON
# ===============================================================================

def train_iris_models(X_train, X_test, y_train, y_test):
    """Train and compare multiple ML algorithms"""

    print("\n🤖 MODEL TRAINING & COMPARISON")
    print("-" * 40)

    # Define models to compare
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }

    results = {}

    # Train and evaluate each model
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

        # Train on full training set
        model.fit(X_train, y_train)

        # Test set predictions
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)

        results[name] = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': test_accuracy,
            'predictions': y_pred
        }

        print(f"{name}:")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print()

    return results


# ===============================================================================
# 5. HYPERPARAMETER OPTIMIZATION
# ===============================================================================

def optimize_best_model(X_train, y_train, X_test, y_test):
    """Optimize the best performing model"""

    print("🎯 HYPERPARAMETER OPTIMIZATION")
    print("-" * 35)

    # Random Forest typically performs well on Iris
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_

    print(f"✓ Best Parameters: {grid_search.best_params_}")
    print(f"✓ Best CV Score: {best_score:.4f}")

    # Final evaluation
    final_predictions = best_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, final_predictions)

    print(f"✓ Final Test Accuracy: {final_accuracy:.4f}")

    return best_model, final_predictions


# ===============================================================================
# 6. MODEL EVALUATION AND VISUALIZATION
# ===============================================================================

def evaluate_iris_performance(y_test, y_pred, model, feature_names, label_encoder):
    """Comprehensive model evaluation"""

    print("\n📊 COMPREHENSIVE MODEL EVALUATION")
    print("-" * 45)

    # Get original species names
    species_names = label_encoder.inverse_transform([0, 1, 2])
    species_names = [name.replace('Iris-', '') for name in species_names]

    # Classification Report
    print("CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=species_names))

    # Confusion Matrix Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=species_names,
                yticklabels=species_names,
                ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    # Feature Importance
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)

        importance_df.plot(x='feature', y='importance', kind='barh', ax=axes[1])
        axes[1].set_title('Feature Importance')
        axes[1].set_xlabel('Importance Score')

    plt.tight_layout()
    plt.show()


# ===============================================================================
# 7. MAIN EXECUTION PIPELINE
# ===============================================================================

def main_iris_classification(dataset_path):
    """Main execution pipeline for Iris classification"""

    # Step 1: Load and explore data
    df = load_iris_data(dataset_path)
    if df is None:
        return None, None, None

    # Step 2: Perform EDA
    perform_iris_eda(df)

    # Step 3: Feature engineering
    df_enhanced = engineer_iris_features(df)

    # Step 4: Prepare data for modeling
    feature_columns = [col for col in df_enhanced.columns if col not in ['Id', 'Species']]
    X = df_enhanced[feature_columns]

    # Encode target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_enhanced['Species'])

    print(f"\n🎯 TARGET ENCODING:")
    for i, species in enumerate(label_encoder.classes_):
        print(f"  {species} → {i}")

    # Step 5: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42, stratify=y)

    # Step 6: Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n📊 DATA SPLIT:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Feature count: {X_train.shape[1]}")

    # Step 7: Model comparison
    results = train_iris_models(X_train_scaled, X_test_scaled, y_train, y_test)

    # Step 8: Optimize best model
    best_model, final_predictions = optimize_best_model(X_train_scaled, y_train,
                                                        X_test_scaled, y_test)

    # Step 9: Final evaluation
    evaluate_iris_performance(y_test, final_predictions, best_model,
                              feature_columns, label_encoder)

    # Step 10: Save model and preprocessing objects
    import os
    os.makedirs('models', exist_ok=True)

    joblib.dump(best_model, 'models/iris_best_model.pkl')
    joblib.dump(scaler, 'models/iris_scaler.pkl')
    joblib.dump(label_encoder, 'models/iris_label_encoder.pkl')

    print("\n✅ IRIS CLASSIFICATION PROJECT COMPLETED!")
    print("📁 Model saved to: models/iris_best_model.pkl")
    print("📁 Scaler saved to: models/iris_scaler.pkl")
    print("📁 Label encoder saved to: models/iris_label_encoder.pkl")

    return best_model, scaler, label_encoder


# ===============================================================================
# 8. PREDICTION FUNCTION FOR NEW DATA
# ===============================================================================

def predict_iris_species(sepal_length, sepal_width, petal_length, petal_width,
                         model_path='models/iris_best_model.pkl',
                         scaler_path='models/iris_scaler.pkl',
                         encoder_path='models/iris_label_encoder.pkl'):
    """Predict iris species for new measurements"""

    try:
        # Load saved models
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(encoder_path)

        # Create feature vector with engineered features
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Engineer features (same as training)
        sepal_ratio = sepal_length / sepal_width
        petal_ratio = petal_length / petal_width
        sepal_area = sepal_length * sepal_width
        petal_area = petal_length * petal_width
        length_ratio = petal_length / sepal_length
        width_ratio = petal_width / sepal_width
        total_area = sepal_area + petal_area
        sepal_perimeter = 2 * (sepal_length + sepal_width)
        petal_perimeter = 2 * (petal_length + petal_width)

        # Combine all features
        all_features = np.array([[sepal_length, sepal_width, petal_length, petal_width,
                                  sepal_ratio, petal_ratio, sepal_area, petal_area,
                                  length_ratio, width_ratio, total_area,
                                  sepal_perimeter, petal_perimeter]])

        # Scale features
        features_scaled = scaler.transform(all_features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        # Get species name
        species = label_encoder.inverse_transform([prediction])[0]

        print(f"\n🌸 PREDICTION RESULTS:")
        print(f"Predicted Species: {species}")
        print(f"Confidence: {probability[prediction]:.2%}")
        print(f"\nAll Probabilities:")
        for i, prob in enumerate(probability):
            sp_name = label_encoder.inverse_transform([i])[0]
            print(f"  {sp_name}: {prob:.2%}")

        return species, probability

    except FileNotFoundError:
        print("❌ Model files not found! Please train the model first.")
        return None, None


# Execute the complete pipeline
if __name__ == "__main__":
    # CHANGE THIS PATH TO YOUR DATASET LOCATION
    DATASET_PATH = "D:\pycharm\code Alpha\Iris.csv"  # Update this path as needed

    print("🚀 Starting Iris Classification Pipeline...")
    print(f"Dataset path: {DATASET_PATH}")

    # Run the complete pipeline
    model, scaler, encoder = main_iris_classification(DATASET_PATH)

    if model is not None:
        print("\n🎯 TESTING PREDICTION FUNCTION:")
        # Test with a sample prediction
        predict_iris_species(5.1, 3.5, 1.4, 0.2)  # Should be Setosa
