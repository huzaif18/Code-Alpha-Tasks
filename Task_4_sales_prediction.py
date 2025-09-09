# ===============================================================================
# CODEALPHA INTERNSHIP - TASK 4: SALES PREDICTION (FINAL FIXED VERSION)
# Author: Huzaif Ali Khan Bepari
# Date: August 2025
# Description: Advanced sales forecasting using advertising spend data
# ===============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

print("💰 CODEALPHA SALES PREDICTION PROJECT")
print("=" * 50)


# ===============================================================================
# 1. DATA LOADING AND EXPLORATION
# ===============================================================================

def load_advertising_data(file_path):
    """Load and explore the advertising dataset"""

    print("📂 LOADING DATASET...")
    print(f"Dataset path: {file_path}")

    try:
        # Load the dataset
        df = pd.read_csv(file_path)

        # Remove the first column if it's an index
        if len(df.columns) > 4 and (df.columns[0] == 'Unnamed: 0' or df.columns[0] == ''):
            df = df.drop(df.columns[0], axis=1)

        print("\n📋 DATASET OVERVIEW")
        print("-" * 30)
        print(f"Dataset Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Missing Values: {df.isnull().sum().sum()}")

        # Display first few rows
        print("\n📊 SAMPLE DATA:")
        print(df.head())

        # Basic statistics
        if 'Sales' in df.columns:
            print(f"\nSales Range: ${df['Sales'].min():.1f}K - ${df['Sales'].max():.1f}K")
            print(f"Average Sales: ${df['Sales'].mean():.1f}K")
            print(f"Total Advertising Spend: ${(df['TV'] + df['Radio'] + df['Newspaper']).sum():.1f}K")

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
# 2. COMPREHENSIVE EDA (FIXED VERSION)
# ===============================================================================

def perform_advertising_eda(df):
    """Comprehensive EDA for sales prediction"""

    print("\n🔍 COMPREHENSIVE ADVERTISING DATA ANALYSIS")
    print("-" * 50)

    # Create a copy for EDA calculations
    df_eda = df.copy()

    # Create comprehensive visualizations
    fig, axes = plt.subplots(4, 3, figsize=(20, 20))
    fig.suptitle('Sales Prediction - Comprehensive Business Analysis', fontsize=16, fontweight='bold')

    # 1. Sales distribution
    df_eda['Sales'].hist(bins=30, ax=axes[0, 0], color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Sales Distribution')
    axes[0, 0].set_xlabel('Sales ($000)')
    axes[0, 0].axvline(df_eda['Sales'].mean(), color='red', linestyle='--',
                       label=f'Mean: ${df_eda["Sales"].mean():.1f}K')
    axes[0, 0].legend()

    # 2. TV advertising vs Sales
    axes[0, 1].scatter(df_eda['TV'], df_eda['Sales'], alpha=0.7, color='blue')
    axes[0, 1].set_title('TV Advertising vs Sales')
    axes[0, 1].set_xlabel('TV Advertising ($000)')
    axes[0, 1].set_ylabel('Sales ($000)')

    # 3. Radio advertising vs Sales
    axes[0, 2].scatter(df_eda['Radio'], df_eda['Sales'], alpha=0.7, color='green')
    axes[0, 2].set_title('Radio Advertising vs Sales')
    axes[0, 2].set_xlabel('Radio Advertising ($000)')
    axes[0, 2].set_ylabel('Sales ($000)')

    # 4. Newspaper advertising vs Sales
    axes[1, 0].scatter(df_eda['Newspaper'], df_eda['Sales'], alpha=0.7, color='orange')
    axes[1, 0].set_title('Newspaper Advertising vs Sales')
    axes[1, 0].set_xlabel('Newspaper Advertising ($000)')
    axes[1, 0].set_ylabel('Sales ($000)')

    # 5. Total advertising spend vs Sales
    df_eda['Total_Advertising'] = df_eda['TV'] + df_eda['Radio'] + df_eda['Newspaper']
    axes[1, 1].scatter(df_eda['Total_Advertising'], df_eda['Sales'], alpha=0.7, color='purple')
    axes[1, 1].set_title('Total Advertising vs Sales')
    axes[1, 1].set_xlabel('Total Advertising ($000)')
    axes[1, 1].set_ylabel('Sales ($000)')

    # 6. Correlation heatmap
    correlation_matrix = df_eda[['TV', 'Radio', 'Newspaper', 'Sales']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[1, 2])
    axes[1, 2].set_title('Advertising Channels Correlation')

    # 7. Advertising channel comparison (box plots)
    ad_data = pd.melt(df_eda[['TV', 'Radio', 'Newspaper']], var_name='Channel', value_name='Spend')
    sns.boxplot(data=ad_data, x='Channel', y='Spend', ax=axes[2, 0])
    axes[2, 0].set_title('Advertising Spend Distribution by Channel')
    axes[2, 0].set_ylabel('Advertising Spend ($000)')

    # 8. Advertising efficiency (Sales per dollar spent)
    df_eda['TV_Efficiency'] = df_eda['Sales'] / (df_eda['TV'] + 1)
    df_eda['Radio_Efficiency'] = df_eda['Sales'] / (df_eda['Radio'] + 1)
    df_eda['Newspaper_Efficiency'] = df_eda['Sales'] / (df_eda['Newspaper'] + 1)

    efficiency_data = pd.DataFrame({
        'TV': df_eda['TV_Efficiency'],
        'Radio': df_eda['Radio_Efficiency'],
        'Newspaper': df_eda['Newspaper_Efficiency']
    })

    efficiency_data.boxplot(ax=axes[2, 1])
    axes[2, 1].set_title('Advertising Efficiency by Channel')
    axes[2, 1].set_ylabel('Sales per Dollar Spent')

    # 9. Advertising mix analysis
    df_eda['TV_Percentage'] = df_eda['TV'] / (df_eda['Total_Advertising'] + 1) * 100
    axes[2, 2].scatter(df_eda['TV_Percentage'], df_eda['Sales'], alpha=0.7, label='TV %')
    axes[2, 2].set_title('TV Percentage of Budget vs Sales')
    axes[2, 2].set_xlabel('TV % of Total Budget')
    axes[2, 2].set_ylabel('Sales ($000)')

    # 10. Sales vs advertising spending ranges - CREATE CATEGORIES FIRST
    def categorize_spend(spend, thresholds=[50, 150]):
        if spend < thresholds[0]:
            return 'Low'
        elif spend < thresholds[1]:
            return 'Medium'
        else:
            return 'High'

    # CREATE THE CATEGORY COLUMNS BEFORE USING THEM
    df_eda['TV_Category'] = df_eda['TV'].apply(lambda x: categorize_spend(x))
    df_eda['Radio_Category'] = df_eda['Radio'].apply(lambda x: categorize_spend(x, [10, 30]))
    df_eda['Newspaper_Category'] = df_eda['Newspaper'].apply(lambda x: categorize_spend(x, [20, 60]))

    # Now the boxplot will work
    df_eda.boxplot(column='Sales', by='TV_Category', ax=axes[3, 0])
    axes[3, 0].set_title('Sales by TV Spending Category')
    axes[3, 0].set_xlabel('TV Spending Level')

    # 11. Residual analysis (for linear relationship)
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    X_simple = df_eda[['TV', 'Radio', 'Newspaper']]
    y = df_eda['Sales']
    lr.fit(X_simple, y)
    y_pred_simple = lr.predict(X_simple)
    residuals = y - y_pred_simple

    axes[3, 1].scatter(y_pred_simple, residuals, alpha=0.7)
    axes[3, 1].axhline(y=0, color='red', linestyle='--')
    axes[3, 1].set_title('Residuals vs Predicted (Linear Model)')
    axes[3, 1].set_xlabel('Predicted Sales')
    axes[3, 1].set_ylabel('Residuals')

    # 12. ROI analysis
    df_eda['Total_ROI'] = (df_eda['Sales'] - df_eda['Total_Advertising']) / (df_eda['Total_Advertising'] + 1) * 100
    df_eda['Total_ROI'].hist(bins=20, ax=axes[3, 2], color='gold', alpha=0.7)
    axes[3, 2].set_title('Return on Investment Distribution')
    axes[3, 2].set_xlabel('ROI (%)')
    axes[3, 2].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # Calculate and display key insights
    print("\n💡 KEY BUSINESS INSIGHTS:")

    # Correlation analysis
    correlations = df_eda[['TV', 'Radio', 'Newspaper']].corrwith(df_eda['Sales']).sort_values(ascending=False)
    print("📊 ADVERTISING EFFECTIVENESS (Correlation with Sales):")
    for channel, corr in correlations.items():
        print(f"  {channel}: {corr:.3f}")

    # Efficiency analysis
    avg_efficiencies = {
        'TV': df_eda['TV_Efficiency'].mean(),
        'Radio': df_eda['Radio_Efficiency'].mean(),
        'Newspaper': df_eda['Newspaper_Efficiency'].mean()
    }
    best_channel = max(avg_efficiencies, key=avg_efficiencies.get)
    print(f"\n🎯 MOST EFFICIENT CHANNEL: {best_channel}")
    print(f"   Average efficiency: {avg_efficiencies[best_channel]:.2f} sales per dollar")

    # ROI analysis
    avg_roi = df_eda['Total_ROI'].mean()
    print(f"\n💰 AVERAGE ROI: {avg_roi:.1f}%")

    # Budget recommendations
    high_performers = df_eda[df_eda['Sales'] > df_eda['Sales'].quantile(0.75)]
    print(f"\n🏆 HIGH-PERFORMING CAMPAIGNS CHARACTERISTICS:")
    print(f"   Average TV spend: ${high_performers['TV'].mean():.1f}K")
    print(f"   Average Radio spend: ${high_performers['Radio'].mean():.1f}K")
    print(f"   Average Newspaper spend: ${high_performers['Newspaper'].mean():.1f}K")


# ===============================================================================
# 3. ADVANCED FEATURE ENGINEERING
# ===============================================================================

def engineer_advertising_features(df, is_training=True):
    """Create advanced features for sales prediction"""

    print("\n🔧 ADVANCED FEATURE ENGINEERING")
    print("-" * 40)

    df_enhanced = df.copy()

    # 1. Basic derived features
    df_enhanced['Total_Advertising'] = df_enhanced['TV'] + df_enhanced['Radio'] + df_enhanced['Newspaper']

    # 2. Interaction features (media synergy effects)
    df_enhanced['TV_Radio_Interaction'] = df_enhanced['TV'] * df_enhanced['Radio']
    df_enhanced['TV_Newspaper_Interaction'] = df_enhanced['TV'] * df_enhanced['Newspaper']
    df_enhanced['Radio_Newspaper_Interaction'] = df_enhanced['Radio'] * df_enhanced['Newspaper']
    df_enhanced['All_Media_Interaction'] = df_enhanced['TV'] * df_enhanced['Radio'] * df_enhanced['Newspaper']

    # 3. Ratio features (budget allocation analysis)
    df_enhanced['TV_Radio_Ratio'] = df_enhanced['TV'] / (df_enhanced['Radio'] + 1)
    df_enhanced['TV_Newspaper_Ratio'] = df_enhanced['TV'] / (df_enhanced['Newspaper'] + 1)
    df_enhanced['Radio_Newspaper_Ratio'] = df_enhanced['Radio'] / (df_enhanced['Newspaper'] + 1)

    # 4. Percentage features (budget distribution)
    df_enhanced['TV_Percentage'] = df_enhanced['TV'] / (df_enhanced['Total_Advertising'] + 1) * 100
    df_enhanced['Radio_Percentage'] = df_enhanced['Radio'] / (df_enhanced['Total_Advertising'] + 1) * 100
    df_enhanced['Newspaper_Percentage'] = df_enhanced['Newspaper'] / (df_enhanced['Total_Advertising'] + 1) * 100

    # 5. Polynomial features for non-linear relationships
    df_enhanced['TV_Squared'] = df_enhanced['TV'] ** 2
    df_enhanced['Radio_Squared'] = df_enhanced['Radio'] ** 2
    df_enhanced['Newspaper_Squared'] = df_enhanced['Newspaper'] ** 2

    # 6. Logarithmic features (diminishing returns effect)
    df_enhanced['TV_Log'] = np.log1p(df_enhanced['TV'])
    df_enhanced['Radio_Log'] = np.log1p(df_enhanced['Radio'])
    df_enhanced['Newspaper_Log'] = np.log1p(df_enhanced['Newspaper'])

    # 7. Categorical features based on spending levels
    def categorize_spend(spend, channel_type):
        if channel_type == 'TV':
            if spend < 50:
                return 'Low'
            elif spend < 150:
                return 'Medium'
            else:
                return 'High'
        elif channel_type == 'Radio':
            if spend < 10:
                return 'Low'
            elif spend < 30:
                return 'Medium'
            else:
                return 'High'
        else:  # Newspaper
            if spend < 20:
                return 'Low'
            elif spend < 60:
                return 'Medium'
            else:
                return 'High'

    df_enhanced['TV_Category'] = df_enhanced['TV'].apply(lambda x: categorize_spend(x, 'TV'))
    df_enhanced['Radio_Category'] = df_enhanced['Radio'].apply(lambda x: categorize_spend(x, 'Radio'))
    df_enhanced['Newspaper_Category'] = df_enhanced['Newspaper'].apply(lambda x: categorize_spend(x, 'Newspaper'))

    # 8. Efficiency metrics - only calculate during training
    if is_training and 'Sales' in df_enhanced.columns:
        df_enhanced['TV_Efficiency'] = df_enhanced['Sales'] / (df_enhanced['TV'] + 1)
        df_enhanced['Radio_Efficiency'] = df_enhanced['Sales'] / (df_enhanced['Radio'] + 1)
        df_enhanced['Newspaper_Efficiency'] = df_enhanced['Sales'] / (df_enhanced['Newspaper'] + 1)
        df_enhanced['Overall_Efficiency'] = df_enhanced['Sales'] / (df_enhanced['Total_Advertising'] + 1)
        # ROI calculation - ONLY during training
        df_enhanced['Total_ROI'] = (df_enhanced['Sales'] - df_enhanced['Total_Advertising']) / (df_enhanced['Total_Advertising'] + 1) * 100
    else:
        # For prediction, set default values
        df_enhanced['TV_Efficiency'] = 0.1
        df_enhanced['Radio_Efficiency'] = 0.5
        df_enhanced['Newspaper_Efficiency'] = 0.3
        df_enhanced['Overall_Efficiency'] = 0.3
        # ROI estimation for prediction
        estimated_sales = 0.05 * df_enhanced['TV'] + 0.18 * df_enhanced['Radio'] + 0.003 * df_enhanced['Newspaper'] + 2.5
        df_enhanced['Total_ROI'] = (estimated_sales - df_enhanced['Total_Advertising']) / (df_enhanced['Total_Advertising'] + 1) * 100

    # 9. Diversification index
    def calculate_diversification_index(row):
        total = row['Total_Advertising']
        if total == 0:
            return 0
        tv_prop = row['TV'] / total
        radio_prop = row['Radio'] / total
        newspaper_prop = row['Newspaper'] / total
        herfindahl = tv_prop ** 2 + radio_prop ** 2 + newspaper_prop ** 2
        return 1 - herfindahl

    df_enhanced['Diversification_Index'] = df_enhanced.apply(calculate_diversification_index, axis=1)

    # 10. Threshold features
    df_enhanced['TV_High_Budget'] = (df_enhanced['TV'] > 150).astype(int)
    df_enhanced['Radio_High_Budget'] = (df_enhanced['Radio'] > 25).astype(int)
    df_enhanced['Newspaper_High_Budget'] = (df_enhanced['Newspaper'] > 30).astype(int)
    df_enhanced['Total_High_Budget'] = (df_enhanced['Total_Advertising'] > 200).astype(int)

    print("✓ Created interaction features (media synergy)")
    print("✓ Created ratio and percentage features")
    print("✓ Created polynomial and logarithmic features")
    print("✓ Created categorical spending levels")
    print("✓ Created efficiency metrics")
    print("✓ Created ROI features")
    print("✓ Created diversification index")
    print("✓ Created threshold indicators")
    print(f"✓ Total features: {len([col for col in df_enhanced.columns if col != 'Sales'])}")

    return df_enhanced


# ===============================================================================
# 4. MODEL TRAINING AND COMPARISON
# ===============================================================================

def train_sales_models(X_train, X_test, y_train, y_test):
    """Train and compare sales prediction models"""

    print("\n🤖 SALES PREDICTION MODEL COMPARISON")
    print("-" * 45)

    # Define models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Lasso Regression': Lasso(alpha=0.1, random_state=42),
        'Elastic Net': ElasticNet(alpha=0.1, random_state=42)
    }

    results = {}

    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

        # For linear models, apply scaling
        if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model.fit(X_train_scaled, y_train)
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

        results[name] = {
            'model': model,
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
        print(f"  Test RMSE: ${test_rmse:.2f}K")
        print(f"  Test MAE: ${test_mae:.2f}K")
        print(f"  MAPE: {mape:.1f}%")
        print()

    return results


# ===============================================================================
# 5. MODEL OPTIMIZATION
# ===============================================================================

def optimize_sales_model(X_train, y_train, X_test, y_test):
    """Optimize the Random Forest model"""

    print("🎯 HYPERPARAMETER OPTIMIZATION")
    print("-" * 35)

    # Parameter grid for Random Forest
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)

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
    print(f"✓ Final Test RMSE: ${final_rmse:.2f}K")
    print(f"✓ Final MAPE: {final_mape:.1f}%")

    return best_model, final_predictions


# ===============================================================================
# 6. BUSINESS INSIGHTS AND RECOMMENDATIONS
# ===============================================================================

def generate_business_insights(df, model, feature_names):
    """Generate actionable business insights"""

    print("\n💼 BUSINESS INSIGHTS & RECOMMENDATIONS")
    print("-" * 50)

    # Feature importance analysis
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)

        print("🎯 TOP 15 SUCCESS FACTORS:")
        for i, row in importance_df.iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")

    # Channel effectiveness analysis
    print("\n📊 CHANNEL EFFECTIVENESS ANALYSIS:")
    correlations = df[['TV', 'Radio', 'Newspaper']].corrwith(df['Sales'])
    for channel, corr in correlations.sort_values(ascending=False).items():
        effectiveness = "High" if corr > 0.7 else "Medium" if corr > 0.4 else "Low"
        print(f"  {channel}: {corr:.3f} ({effectiveness} effectiveness)")

    # ROI analysis
    df_temp = df.copy()
    df_temp['Total_Advertising'] = df_temp['TV'] + df_temp['Radio'] + df_temp['Newspaper']
    df_temp['ROI'] = (df_temp['Sales'] - df_temp['Total_Advertising']) / df_temp['Total_Advertising'] * 100

    print(f"\n💰 ROI ANALYSIS:")
    print(f"  Average ROI: {df_temp['ROI'].mean():.1f}%")
    print(f"  Best ROI: {df_temp['ROI'].max():.1f}%")
    print(f"  ROI Standard Deviation: {df_temp['ROI'].std():.1f}%")

    # Budget optimization recommendations
    high_performers = df[df['Sales'] > df['Sales'].quantile(0.8)]

    print(f"\n🏆 HIGH-PERFORMING CAMPAIGNS (Top 20%):")
    print(f"  Average TV spend: ${high_performers['TV'].mean():.1f}K")
    print(f"  Average Radio spend: ${high_performers['Radio'].mean():.1f}K")
    print(f"  Average Newspaper spend: ${high_performers['Newspaper'].mean():.1f}K")
    print(f"  Average Total spend: ${(high_performers['TV'] + high_performers['Radio'] + high_performers['Newspaper']).mean():.1f}K")
    print(f"  Average Sales: ${high_performers['Sales'].mean():.1f}K")

    # Optimal budget allocation
    print(f"\n📈 RECOMMENDED BUDGET ALLOCATION:")
    total_optimal = (high_performers['TV'] + high_performers['Radio'] + high_performers['Newspaper']).mean()
    tv_percent = high_performers['TV'].mean() / total_optimal * 100
    radio_percent = high_performers['Radio'].mean() / total_optimal * 100
    newspaper_percent = high_performers['Newspaper'].mean() / total_optimal * 100

    print(f"  TV: {tv_percent:.1f}% of budget")
    print(f"  Radio: {radio_percent:.1f}% of budget")
    print(f"  Newspaper: {newspaper_percent:.1f}% of budget")


# ===============================================================================
# 7. MODEL EVALUATION
# ===============================================================================

def evaluate_sales_model_performance(y_test, y_pred, model, feature_names):
    """Evaluate sales prediction model performance"""

    print("\n📊 SALES MODEL EVALUATION")
    print("-" * 35)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: ${rmse:.2f}K")
    print(f"MAE: ${mae:.2f}K")
    print(f"MAPE: {mape:.1f}%")

    # Create evaluation visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Sales Prediction Model - Performance Evaluation', fontsize=16)

    # 1. Actual vs Predicted
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='blue')
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Sales ($000)')
    axes[0, 0].set_ylabel('Predicted Sales ($000)')
    axes[0, 0].set_title(f'Actual vs Predicted (R² = {r2:.3f})')

    # 2. Residuals plot
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Sales ($000)')
    axes[0, 1].set_ylabel('Residuals ($000)')
    axes[0, 1].set_title('Residual Analysis')

    # 3. Residual distribution
    axes[1, 0].hist(residuals, bins=20, alpha=0.7, color='skyblue')
    axes[1, 0].set_xlabel('Residuals ($000)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residual Distribution')

    # 4. Feature importance
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)

        importance_df.plot(x='feature', y='importance', kind='barh', ax=axes[1, 1])
        axes[1, 1].set_title('Top 10 Feature Importances')
        axes[1, 1].set_xlabel('Importance Score')

    plt.tight_layout()
    plt.show()


# ===============================================================================
# 8. MAIN EXECUTION PIPELINE
# ===============================================================================

def main_sales_prediction(dataset_path):
    """Main execution pipeline for sales prediction"""

    # Step 1: Load data
    df = load_advertising_data(dataset_path)
    if df is None:
        return None, None

    # Step 2: Comprehensive EDA
    perform_advertising_eda(df)

    # Step 3: Advanced feature engineering (training mode)
    df_enhanced = engineer_advertising_features(df, is_training=True)

    # Step 4: Prepare features (exclude categorical columns for now)
    exclude_columns = ['Sales', 'TV_Category', 'Radio_Category', 'Newspaper_Category']
    feature_columns = [col for col in df_enhanced.columns if col not in exclude_columns]

    X = df_enhanced[feature_columns]
    y = df_enhanced['Sales']

    print(f"\n📊 DATA PREPARATION:")
    print(f"Feature set size: {X.shape}")
    print(f"Target variable range: ${y.min():.1f}K - ${y.max():.1f}K")

    # Step 5: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Step 6: Model training and comparison
    results = train_sales_models(X_train, X_test, y_train, y_test)

    # Step 7: Model optimization
    best_model, final_predictions = optimize_sales_model(X_train, y_train,
                                                         X_test, y_test)

    # Step 8: Model evaluation
    evaluate_sales_model_performance(y_test, final_predictions, best_model, feature_columns)

    # Step 9: Business insights
    generate_business_insights(df_enhanced, best_model, feature_columns)

    # Step 10: Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/sales_prediction_model.pkl')

    print("\n✅ SALES PREDICTION PROJECT COMPLETED!")
    print("📁 Model saved to: models/sales_prediction_model.pkl")

    return best_model, results


# ===============================================================================
# 9. PREDICTION FUNCTION
# ===============================================================================

def predict_sales(tv_spend, radio_spend, newspaper_spend,
                  model_path='models/sales_prediction_model.pkl'):
    """Predict sales for new advertising spend"""

    try:
        # Load saved model
        model = joblib.load(model_path)

        # Create feature dataframe
        data = {
            'TV': [tv_spend],
            'Radio': [radio_spend],
            'Newspaper': [newspaper_spend]
        }

        df = pd.DataFrame(data)

        # Engineer features (prediction mode)
        df_enhanced = engineer_advertising_features(df, is_training=False)

        # Prepare features (exclude categorical columns)
        exclude_columns = ['Sales', 'TV_Category', 'Radio_Category', 'Newspaper_Category']
        feature_columns = [col for col in df_enhanced.columns if col not in exclude_columns]
        X = df_enhanced[feature_columns]

        # Make prediction
        prediction = model.predict(X)[0]

        # Calculate metrics
        total_spend = tv_spend + radio_spend + newspaper_spend
        roi = (prediction - total_spend) / total_spend * 100 if total_spend > 0 else 0

        print(f"\n💰 SALES PREDICTION RESULTS:")
        print(f"TV Advertising: ${tv_spend}K")
        print(f"Radio Advertising: ${radio_spend}K")
        print(f"Newspaper Advertising: ${newspaper_spend}K")
        print(f"Total Advertising Spend: ${total_spend}K")
        print(f"Predicted Sales: ${prediction:.1f}K")
        print(f"Expected ROI: {roi:.1f}%")
        print(f"Expected Profit: ${prediction - total_spend:.1f}K")

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
    # Use the attached file directly (fixed path)
    DATASET_PATH = r"D:\pycharm\code Alpha\Advertising.csv"  # Fixed with raw string

    print("🚀 Starting Sales Prediction Pipeline...")
    print(f"Dataset path: {DATASET_PATH}")

    # Run the complete pipeline
    model, results = main_sales_prediction(DATASET_PATH)

    if model is not None:
        print("\n🎯 TESTING PREDICTION FUNCTION:")
        # Test with sample predictions
        predict_sales(200, 35, 40)  # High budget scenario
        predict_sales(100, 20, 25)  # Medium budget scenario
        predict_sales(50, 10, 15)   # Low budget scenario
