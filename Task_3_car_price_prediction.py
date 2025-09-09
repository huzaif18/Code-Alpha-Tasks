# ===============================================================================
# CODEALPHA INTERNSHIP - TASK 3: CAR PRICE PREDICTION (FIXED VERSION)
# Author: Huzaif Ali Khan Bepari
# Date: August 2025
# Description: Professional ML regression model for car price prediction
# ===============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

print("🚗 CODEALPHA CAR PRICE PREDICTION PROJECT")
print("=" * 50)


# ===============================================================================
# 1. DATA LOADING AND EXPLORATION
# ===============================================================================

def load_car_data(file_path):
    """Load and explore the car dataset"""

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

        # Basic statistics
        if 'Selling_Price' in df.columns:
            print(f"\nPrice Range: ₹{df['Selling_Price'].min():.2f} - ₹{df['Selling_Price'].max():.2f} Lakhs")
            print(f"Average Price: ₹{df['Selling_Price'].mean():.2f} Lakhs")

        print(f"Year Range: {df['Year'].min()} - {df['Year'].max()}")

        # Data types
        print("\n📝 DATA TYPES:")
        print(df.dtypes)

        return df

    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found!")
        print("Please check the file path and try again.")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return None


# ===============================================================================
# 2. COMPREHENSIVE EDA FOR CAR DATA
# ===============================================================================

def perform_car_eda(df):
    """Comprehensive EDA for car price prediction"""

    print("\n🔍 COMPREHENSIVE CAR DATA ANALYSIS")
    print("-" * 45)

    # Create comprehensive visualization
    fig, axes = plt.subplots(4, 3, figsize=(20, 20))
    fig.suptitle('Car Price Prediction - Comprehensive EDA', fontsize=16, fontweight='bold')

    # 1. Price distribution
    df['Selling_Price'].hist(bins=50, ax=axes[0, 0], color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Selling Price Distribution')
    axes[0, 0].set_xlabel('Price (₹ Lakhs)')
    axes[0, 0].axvline(df['Selling_Price'].mean(), color='red', linestyle='--',
                       label=f'Mean: ₹{df["Selling_Price"].mean():.2f}L')
    axes[0, 0].legend()

    # 2. Price by fuel type
    df.boxplot(column='Selling_Price', by='Fuel_Type', ax=axes[0, 1])
    axes[0, 1].set_title('Price by Fuel Type')
    axes[0, 1].set_xlabel('Fuel Type')

    # 3. Price by transmission
    df.boxplot(column='Selling_Price', by='Transmission', ax=axes[0, 2])
    axes[0, 2].set_title('Price by Transmission')
    axes[0, 2].set_xlabel('Transmission')

    # 4. Car age vs Price
    df['Car_Age'] = 2024 - df['Year']
    axes[1, 0].scatter(df['Car_Age'], df['Selling_Price'], alpha=0.6, color='green')
    axes[1, 0].set_title('Car Age vs Selling Price')
    axes[1, 0].set_xlabel('Car Age (years)')
    axes[1, 0].set_ylabel('Selling Price (₹ Lakhs)')

    # 5. Driven_kms vs Price
    axes[1, 1].scatter(df['Driven_kms'], df['Selling_Price'], alpha=0.6, color='purple')
    axes[1, 1].set_title('Kilometers Driven vs Price')
    axes[1, 1].set_xlabel('Driven KMs')
    axes[1, 1].set_ylabel('Selling Price (₹ Lakhs)')

    # 6. Present Price vs Selling Price
    axes[1, 2].scatter(df['Present_Price'], df['Selling_Price'], alpha=0.6, color='orange')
    axes[1, 2].set_title('Present Price vs Selling Price')
    axes[1, 2].set_xlabel('Present Price (₹ Lakhs)')
    axes[1, 2].set_ylabel('Selling Price (₹ Lakhs)')

    # 7. Year distribution
    df['Year'].hist(bins=20, ax=axes[2, 0], color='coral')
    axes[2, 0].set_title('Car Year Distribution')
    axes[2, 0].set_xlabel('Year')

    # 8. Top car brands by count
    top_brands = df['Car_Name'].str.split().str[0].value_counts().head(10)
    top_brands.plot(kind='bar', ax=axes[2, 1], color='lightgreen')
    axes[2, 1].set_title('Top 10 Car Brands by Count')
    axes[2, 1].set_xlabel('Brand')
    axes[2, 1].tick_params(axis='x', rotation=45)

    # 9. Seller type distribution
    df['Selling_type'].value_counts().plot(kind='pie', ax=axes[2, 2], autopct='%1.1f%%')
    axes[2, 2].set_title('Seller Type Distribution')

    # 10. Owner type distribution
    df['Owner'].value_counts().plot(kind='bar', ax=axes[3, 0], color='gold')
    axes[3, 0].set_title('Owner Type Distribution')
    axes[3, 0].set_xlabel('Number of Previous Owners')

    # 11. Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[3, 1])
    axes[3, 1].set_title('Feature Correlation Matrix')

    # 12. Price depreciation analysis
    price_depreciation = df.groupby('Car_Age')['Selling_Price'].mean()
    price_depreciation.plot(kind='line', ax=axes[3, 2], marker='o', color='red')
    axes[3, 2].set_title('Average Price by Car Age')
    axes[3, 2].set_xlabel('Car Age (years)')
    axes[3, 2].set_ylabel('Average Selling Price (₹ Lakhs)')

    plt.tight_layout()
    plt.show()

    # Key insights
    print("\n💡 KEY INSIGHTS:")
    print(f"✓ Dataset contains {df.shape[0]} car records")
    print(f"✓ Price range: ₹{df['Selling_Price'].min():.2f}L to ₹{df['Selling_Price'].max():.2f}L")
    print(f"✓ Most cars are {df['Fuel_Type'].mode()[0]} fuel type")
    print(f"✓ {df['Transmission'].mode()[0]} transmission is more common")
    print(f"✓ Strong correlation between Present_Price and Selling_Price")
    print(f"✓ Car age shows clear depreciation pattern")


# ===============================================================================
# 3. ADVANCED FEATURE ENGINEERING (FIXED VERSION)
# ===============================================================================

def engineer_car_features(df, is_training=True):
    """Create advanced features for better price prediction"""

    print("\n🔧 ADVANCED FEATURE ENGINEERING")
    print("-" * 40)

    df_enhanced = df.copy()

    # 1. Car age and basic features
    df_enhanced['Car_Age'] = 2024 - df_enhanced['Year']

    # 2. Depreciation rate - only calculate during training
    if is_training and 'Selling_Price' in df_enhanced.columns:
        df_enhanced['Depreciation_Rate'] = (df_enhanced['Present_Price'] - df_enhanced['Selling_Price']) / df_enhanced[
            'Present_Price']
        df_enhanced['Depreciation_Rate'] = df_enhanced['Depreciation_Rate'].clip(0, 1)
    else:
        # For prediction, set a default value (will be handled by preprocessing)
        df_enhanced['Depreciation_Rate'] = 0.3  # Average depreciation rate

    # 3. Mileage features
    df_enhanced['Kms_per_Year'] = df_enhanced['Driven_kms'] / (df_enhanced['Car_Age'] + 1)
    df_enhanced['High_Mileage'] = (df_enhanced['Driven_kms'] > 50000).astype(int)  # Fixed threshold

    # 4. Brand extraction and categorization
    df_enhanced['Brand'] = df_enhanced['Car_Name'].str.split().str[0].str.lower()

    # Categorize brands by luxury level
    luxury_brands = ['bmw', 'mercedes', 'audi', 'jaguar', 'lexus', 'volvo']
    premium_brands = ['honda', 'toyota', 'hyundai', 'ford', 'volkswagen']

    def categorize_brand(brand):
        if brand in luxury_brands:
            return 'Luxury'
        elif brand in premium_brands:
            return 'Premium'
        else:
            return 'Economy'

    df_enhanced['Brand_Category'] = df_enhanced['Brand'].apply(categorize_brand)

    # 5. Price efficiency features
    df_enhanced['Price_per_Year'] = df_enhanced['Present_Price'] / (df_enhanced['Car_Age'] + 1)

    # Value retention - only during training
    if is_training and 'Selling_Price' in df_enhanced.columns:
        df_enhanced['Value_Retention'] = df_enhanced['Selling_Price'] / df_enhanced['Present_Price']
    else:
        # For prediction, estimate based on age
        df_enhanced['Value_Retention'] = np.maximum(0.1, 1 - (df_enhanced['Car_Age'] * 0.1))

    # 6. Categorical feature encoding
    df_enhanced['Is_Automatic'] = (df_enhanced['Transmission'] == 'Automatic').astype(int)
    df_enhanced['Is_Diesel'] = (df_enhanced['Fuel_Type'] == 'Diesel').astype(int)
    df_enhanced['Is_Dealer'] = (df_enhanced['Selling_type'] == 'Dealer').astype(int)
    df_enhanced['Is_First_Owner'] = (df_enhanced['Owner'] == 0).astype(int)

    # 7. Age categories
    def age_category(age):
        if age <= 2:
            return 'New'
        elif age <= 5:
            return 'Recent'
        elif age <= 10:
            return 'Mature'
        else:
            return 'Old'

    df_enhanced['Age_Category'] = df_enhanced['Car_Age'].apply(age_category)

    # 8. Interaction features
    df_enhanced['Brand_Age_Interaction'] = df_enhanced['Car_Age'] * (df_enhanced['Brand_Category'] == 'Luxury').astype(
        int)
    df_enhanced['Mileage_Age_Interaction'] = df_enhanced['Driven_kms'] * df_enhanced['Car_Age']

    print("✓ Created age and depreciation features")
    print("✓ Created mileage and usage patterns")
    print("✓ Created brand categorization")
    print("✓ Created price efficiency metrics")
    print("✓ Created categorical encodings")
    print("✓ Created interaction features")
    print(f"✓ Total features: {len(df_enhanced.columns) - 1}")

    return df_enhanced


# ===============================================================================
# 4. DATA PREPROCESSING PIPELINE
# ===============================================================================

def create_car_preprocessing_pipeline():
    """Create preprocessing pipeline for mixed data types"""

    # Categorical features for one-hot encoding
    categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission',
                            'Brand_Category', 'Age_Category']

    # Numerical features for scaling
    numerical_features = ['Year', 'Present_Price', 'Driven_kms', 'Owner', 'Car_Age',
                          'Depreciation_Rate', 'Kms_per_Year', 'High_Mileage',
                          'Price_per_Year', 'Value_Retention', 'Is_Automatic',
                          'Is_Diesel', 'Is_Dealer', 'Is_First_Owner',
                          'Brand_Age_Interaction', 'Mileage_Age_Interaction']

    # Create preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )

    return preprocessor, numerical_features, categorical_features


# ===============================================================================
# 5. MODEL TRAINING AND COMPARISON
# ===============================================================================

def train_car_models(X_train, X_test, y_train, y_test):
    """Train and compare multiple regression models"""

    print("\n🤖 CAR PRICE MODEL COMPARISON")
    print("-" * 40)

    # Get preprocessing pipeline
    preprocessor, _, _ = create_car_preprocessing_pipeline()

    # Define models
    models = {
        'Random Forest': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        'Gradient Boosting': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ]),
        'Linear Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ]),
        'Ridge Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', Ridge(alpha=1.0, random_state=42))
        ]),
        'Lasso Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', Lasso(alpha=0.1, random_state=42))
        ])
    }

    results = {}

    for name, pipeline in models.items():
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train,
                                    cv=5, scoring='r2')

        # Train on full training set
        pipeline.fit(X_train, y_train)

        # Predictions
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)

        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

        results[name] = {
            'model': pipeline,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'mape': mape,
            'predictions': y_pred_test
        }

        print(f"{name}:")
        print(f"  CV R² Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"  Test R² Score: {test_r2:.4f}")
        print(f"  Test RMSE: ₹{test_rmse:.2f}L")
        print(f"  Test MAE: ₹{test_mae:.2f}L")
        print(f"  MAPE: {mape:.1f}%")
        print()

    return results


# ===============================================================================
# 6. MODEL OPTIMIZATION
# ===============================================================================

def optimize_car_model(X_train, y_train, X_test, y_test):
    """Optimize the Random Forest model"""

    print("🎯 HYPERPARAMETER OPTIMIZATION")
    print("-" * 35)

    preprocessor, _, _ = create_car_preprocessing_pipeline()

    # Parameter grid for Random Forest
    param_grid = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [10, 15, 20, None],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    }

    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    # Grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5,
                               scoring='r2', n_jobs=-1, verbose=1)

    print("Performing grid search...")
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_

    print(f"✓ Best Parameters: {grid_search.best_params_}")
    print(f"✓ Best CV R² Score: {best_score:.4f}")

    # Final evaluation
    final_predictions = best_model.predict(X_test)
    final_r2 = r2_score(y_test, final_predictions)
    final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
    final_mape = np.mean(np.abs((y_test - final_predictions) / y_test)) * 100

    print(f"✓ Final Test R² Score: {final_r2:.4f}")
    print(f"✓ Final Test RMSE: ₹{final_rmse:.2f}L")
    print(f"✓ Final MAPE: {final_mape:.1f}%")

    return best_model, final_predictions


# ===============================================================================
# 7. MODEL EVALUATION
# ===============================================================================

def evaluate_car_model_performance(y_test, y_pred, model):
    """Evaluate car price prediction model performance"""

    print("\n📊 FINAL MODEL EVALUATION")
    print("-" * 35)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: ₹{rmse:.2f} Lakhs")
    print(f"MAE: ₹{mae:.2f} Lakhs")
    print(f"MAPE: {mape:.1f}%")
    print(f"Mean Absolute Error %: {(mae / np.mean(y_test) * 100):.1f}%")

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Car Price Prediction - Model Evaluation', fontsize=16)

    # 1. Actual vs Predicted
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Price (₹ Lakhs)')
    axes[0, 0].set_ylabel('Predicted Price (₹ Lakhs)')
    axes[0, 0].set_title(f'Actual vs Predicted (R² = {r2:.3f})')

    # 2. Residuals
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Price (₹ Lakhs)')
    axes[0, 1].set_ylabel('Residuals (₹ Lakhs)')
    axes[0, 1].set_title('Residual Plot')

    # 3. Residual distribution
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='skyblue')
    axes[1, 0].set_xlabel('Residuals (₹ Lakhs)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residual Distribution')

    # 4. Prediction error distribution
    prediction_errors = np.abs(residuals)
    axes[1, 1].hist(prediction_errors, bins=30, alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('Absolute Prediction Error (₹ Lakhs)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Error Distribution')

    plt.tight_layout()
    plt.show()


# ===============================================================================
# 8. MAIN EXECUTION PIPELINE
# ===============================================================================

def main_car_prediction(dataset_path):
    """Main execution pipeline for car price prediction"""

    # Step 1: Load data
    df = load_car_data(dataset_path)
    if df is None:
        return None, None

    # Step 2: EDA
    perform_car_eda(df)

    # Step 3: Feature engineering (training mode)
    df_enhanced = engineer_car_features(df, is_training=True)

    # Step 4: Prepare features and target
    feature_columns = [col for col in df_enhanced.columns
                       if col not in ['Car_Name', 'Selling_Price', 'Brand']]
    X = df_enhanced[feature_columns]
    y = df_enhanced['Selling_Price']

    print(f"\n📊 DATA PREPARATION:")
    print(f"Feature set size: {X.shape}")
    print(f"Target variable range: ₹{y.min():.2f}L - ₹{y.max():.2f}L")

    # Step 5: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Step 6: Model comparison
    results = train_car_models(X_train, X_test, y_train, y_test)

    # Step 7: Optimize best model
    best_model, final_predictions = optimize_car_model(X_train, y_train,
                                                       X_test, y_test)

    # Step 8: Final evaluation
    evaluate_car_model_performance(y_test, final_predictions, best_model)

    # Step 9: Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/car_price_model.pkl')

    print("\n✅ CAR PRICE PREDICTION PROJECT COMPLETED!")
    print("📁 Model saved to: models/car_price_model.pkl")

    return best_model, results


# ===============================================================================
# 9. PREDICTION FUNCTION (FIXED VERSION)
# ===============================================================================

def predict_car_price(car_name, year, present_price, driven_kms, fuel_type,
                      selling_type, transmission, owner,
                      model_path='models/car_price_model.pkl'):
    """Predict car price for new car details"""

    try:
        # Load saved model
        model = joblib.load(model_path)

        # Create feature dataframe
        data = {
            'Car_Name': [car_name],
            'Year': [year],
            'Present_Price': [present_price],
            'Driven_kms': [driven_kms],
            'Fuel_Type': [fuel_type],
            'Selling_type': [selling_type],
            'Transmission': [transmission],
            'Owner': [owner]
        }

        df = pd.DataFrame(data)

        # Engineer features (prediction mode)
        df_enhanced = engineer_car_features(df, is_training=False)

        # Prepare features
        feature_columns = [col for col in df_enhanced.columns
                           if col not in ['Car_Name', 'Selling_Price', 'Brand']]
        X = df_enhanced[feature_columns]

        # Make prediction
        prediction = model.predict(X)[0]

        print(f"\n🚗 CAR PRICE PREDICTION:")
        print(f"Car: {car_name} ({year})")
        print(f"Present Price: ₹{present_price}L")
        print(f"Predicted Selling Price: ₹{prediction:.2f}L")
        print(f"Estimated Depreciation: {((present_price - prediction) / present_price * 100):.1f}%")

        return prediction

    except FileNotFoundError:
        print("❌ Model file not found! Please train the model first.")
        return None
    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")
        return None


# ===============================================================================
# 10. EXECUTE THE COMPLETE PIPELINE
# ===============================================================================

if __name__ == "__main__":
    # CHANGE THIS PATH TO YOUR DATASET LOCATION
    DATASET_PATH = "D:\pycharm\code Alpha\car data.csv"  # Update this path as needed

    print("🚀 Starting Car Price Prediction Pipeline...")
    print(f"Dataset path: {DATASET_PATH}")

    # Run the complete pipeline
    model, results = main_car_prediction(DATASET_PATH)

    if model is not None:
        print("\n🎯 TESTING PREDICTION FUNCTION:")
        # Test with a sample prediction
        predict_car_price("swift", 2014, 6.87, 42450, "Diesel", "Dealer", "Manual", 0)

        print("\n🔮 Additional Test Predictions:")
        predict_car_price("i20", 2016, 8.5, 35000, "Petrol", "Dealer", "Automatic", 0)
        predict_car_price("city", 2018, 12.0, 25000, "Petrol", "Individual", "Manual", 0)
