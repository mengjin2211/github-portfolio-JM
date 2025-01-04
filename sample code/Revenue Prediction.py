import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Create dataset with more realistic price distributions
np.random.seed(42)  # For reproducibility

data = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=365),
    'region': np.random.choice(['Lower Mainland', 'Vancouver Island', 'Interior',
                              'Northern BC', 'Fraser Valley'], 365),
    'zoning': np.random.choice(['Residential', 'Commercial', 'Agricultural',
                              'Industrial', 'Mixed-Use'], 365),
    'lot_size_acres': np.random.lognormal(1, 0.5, 365),
    'waterfront': np.random.choice([0, 1], size=365, p=[0.85, 0.15]),
    'ALR_protected': np.random.choice([0, 1], size=365, p=[0.7, 0.3]),
    'flood_risk': np.random.choice(['Low', 'Medium', 'High'], 365, p=[0.6, 0.3, 0.1]),
    'slope_percentage': np.random.uniform(0, 45, 365),
    'distance_to_highway_km': np.random.lognormal(2, 0.5, 365),
    'services_available': np.random.choice([0, 1], size=365, p=[0.2, 0.8]),
    'avg_neighborhood_price_sqft': np.random.normal(350, 50, 365),
    'days_on_market': np.random.lognormal(3, 0.5, 365),
    'interest_rate': np.random.normal(5, 0.5, 365),
    'subdivision_potential': np.random.choice([0, 1], size=365, p=[0.8, 0.2]),
    'development_restrictions': np.random.choice(['None', 'Moderate', 'Significant'], 365)
})

# Create more realistic base prices (in thousands of dollars)
base_price = np.random.lognormal(6, 0.5, 365) * 1000  # This gives a more realistic range

# Add correlations with proper scaling
data['price_per_acre'] = base_price.copy()
data.loc[data['region'] == 'Lower Mainland', 'price_per_acre'] *= 2.5
data.loc[data['region'] == 'Fraser Valley', 'price_per_acre'] *= 1.8
data.loc[data['waterfront'] == 1, 'price_per_acre'] *= 1.7
data.loc[data['ALR_protected'] == 1, 'price_per_acre'] *= 0.6
data.loc[data['services_available'] == 1, 'price_per_acre'] *= 1.4
data.loc[data['zoning'] == 'Commercial', 'price_per_acre'] *= 1.6
data.loc[data['zoning'] == 'Industrial', 'price_per_acre'] *= 1.4
data.loc[data['flood_risk'] == 'High', 'price_per_acre'] *= 0.7

class BCLandPredictionModel:
    def __init__(self):
        self.numeric_features = ['lot_size_acres', 'distance_to_highway_km',
                               'avg_neighborhood_price_sqft', 'days_on_market',
                               'interest_rate', 'slope_percentage']
        self.categorical_features = ['region', 'zoning', 'flood_risk',
                                   'development_restrictions']
        self.binary_features = ['waterfront', 'ALR_protected',
                              'services_available', 'subdivision_potential']

        # Create preprocessing steps
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False))
        ])
        categorical_transformer = OneHotEncoder(drop='first', sparse=False)

        # Include binary features in the preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features),
                ('bin', 'passthrough', self.binary_features)
            ])

        # Use XGBoost Regressor
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', XGBRegressor(
                random_state=42
            ))
        ])

    def prepare_features(self, df):
        """Prepare features for modeling."""
        df_processed = df.copy()

        # Convert binary columns to int
        for col in self.binary_features:
            df_processed[col] = df_processed[col].astype(int)

        # Log transform lot size (common in real estate modeling)
        df_processed['lot_size_acres'] = np.log1p(df_processed['lot_size_acres'])

        return df_processed

    def train(self, X, y):
        """Train the model with log-transformed target."""
        print("Training model...")
        # Log transform the target variable for better prediction
        self.model.fit(X, np.log1p(y))
        print("Training completed!")

    def predict(self, X):
        """Make predictions and transform back to original scale."""
        log_pred = self.model.predict(X)
        return np.expm1(log_pred)

    def evaluate(self, X, y):
        """Evaluate model performance."""
        y_pred = self.predict(X)

        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        mape = np.mean(np.abs((y - y_pred) / y)) * 100

        # Cross-validation on log-transformed target
        log_y = np.log1p(y)
        cv_scores = cross_val_score(self.model, X, log_y, cv=5, scoring='r2')

        return {
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'CV_scores': cv_scores.mean(),
            'CV_std': cv_scores.std()
        }

    def plot_predictions(self, X, y, plot_type='scatter'):
        """Visualize predictions vs actual values."""
        y_pred = self.predict(X)

        if plot_type == 'scatter':
            plt.figure(figsize=(10, 6))
            plt.scatter(y/1000, y_pred/1000, alpha=0.5)
            plt.plot([y.min()/1000, y.max()/1000],
                    [y.min()/1000, y.max()/1000], 'r--', lw=2)
            plt.xlabel('Actual Price per Acre (Thousands $)')
            plt.ylabel('Predicted Price per Acre (Thousands $)')
            plt.title('Actual vs Predicted Land Prices')

        elif plot_type == 'residuals':
            residuals = y - y_pred
            plt.figure(figsize=(10, 6))
            plt.scatter(y_pred/1000, residuals/1000, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Price per Acre (Thousands $)')
            plt.ylabel('Residuals (Thousands $)')
            plt.title('Residual Plot')

        plt.tight_layout()
        plt.show()

    def hyperparameter_tuning(self, X, y):
        """Perform hyperparameter tuning using Randomized Search."""
        param_dist = {
            'regressor__n_estimators': [300, 500, 700],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__max_depth': [3, 5, 7],
            'regressor__min_child_weight': [1, 3, 5],
            'regressor__subsample': [0.7, 0.8, 0.9],
            'regressor__colsample_bytree': [0.7, 0.8, 0.9]
        }

        random_search = RandomizedSearchCV(self.model, param_distributions=param_dist,
                                          n_iter=50, cv=5, scoring='r2', n_jobs=-1, random_state=42)
        random_search.fit(X, np.log1p(y))

        print("Best parameters found: ", random_search.best_params_)
        self.model = random_search.best_estimator_

if __name__ == "__main__":
    # Initialize model
    model = BCLandPredictionModel()

    # Prepare data
    processed_data = model.prepare_features(data)

    # Select features for training
    features = (model.numeric_features + model.categorical_features +
               model.binary_features)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        processed_data[features],
        processed_data['price_per_acre'],
        test_size=0.2,
        random_state=42
    )

    # Hyperparameter tuning
    model.hyperparameter_tuning(X_train, y_train)

    # Train model
    model.train(X_train, y_train)

    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    print("\nModel Performance:")
    print(f"RMSE: \\\${metrics['RMSE']:,.2f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print(f"R2 Score: {metrics['R2']:.3f}")
    print(f"Cross-validation Score: {metrics['CV_scores']:.3f} (±{metrics['CV_std']:.3f})")

    # Plot predictions
    model.plot_predictions(X_test, y_test, plot_type='scatter')
    model.plot_predictions(X_test, y_test, plot_type='residuals')