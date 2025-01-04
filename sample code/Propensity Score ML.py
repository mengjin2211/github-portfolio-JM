import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

# Generate synthetic donor data
np.random.seed(42)

def generate_donor_data(n_samples=1000):
    data = {
        'alumni_id': range(1, n_samples + 1),
        'years_since_graduation': np.random.randint(1, 50, n_samples),
        'five_year_donations': np.random.exponential(1000, n_samples),
        'largest_gift': np.random.exponential(2000, n_samples),
        'lifetime_giving': np.random.exponential(5000, n_samples),
        'wealth_score': np.random.randint(1, 10, n_samples),
        'event_attendance_rate': np.random.uniform(0, 1, n_samples),
        'volunteer_hours': np.random.exponential(20, n_samples),
        'email_engagement_score': np.random.uniform(0, 1, n_samples),
        'degree_level': np.random.choice(['Bachelors', 'Masters', 'PhD'], n_samples),
        'last_donation_months_ago': np.random.exponential(12, n_samples),
        'total_campaign_contacts': np.random.poisson(5, n_samples)
    }
    
    # Create target variable based on feature combinations
    features_influence = (
        0.3 * (data['wealth_score'] / 10) +
        0.2 * (np.minimum(data['five_year_donations'], 5000) / 5000) +
        0.15 * data['event_attendance_rate'] +
        0.15 * data['email_engagement_score'] +
        0.1 * (np.minimum(data['lifetime_giving'], 10000) / 10000) +
        0.1 * (1 - data['last_donation_months_ago'] / 36)  # Higher score for recent donors
    )
    
    # Convert to binary target with some randomness
    data['donor_probability'] = features_influence
    data['will_donate'] = (features_influence + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)
    
    return pd.DataFrame(data)

# Generate and preprocess data
def prepare_data(df):
    # Create features
    df['giving_frequency'] = df['five_year_donations'] / (df['lifetime_giving'] + 1)
    df['avg_gift_size'] = df['lifetime_giving'] / (df['total_campaign_contacts'] + 1)
    
    # Encode categorical variables
    df = pd.get_dummies(df, columns=['degree_level'])
    
    # Select features for modeling
    feature_columns = [
        'years_since_graduation', 'five_year_donations', 'largest_gift',
        'lifetime_giving', 'wealth_score', 'event_attendance_rate',
        'volunteer_hours', 'email_engagement_score', 'last_donation_months_ago',
        'total_campaign_contacts', 'giving_frequency', 'avg_gift_size',
        'degree_level_Bachelors', 'degree_level_Masters', 'degree_level_PhD'
    ]
    
    return df[feature_columns], df['will_donate']

# Train model and generate propensity scores
def train_and_predict(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Generate predictions and probabilities
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, scaler, y_test, y_pred, y_prob, feature_importance

# Main execution
df = generate_donor_data(1000)
X, y = prepare_data(df)
model, scaler, y_test, y_pred, y_prob, feature_importance = train_and_predict(X, y)

# Print model performance
print("\nModel Performance:")
print(classification_report(y_test, y_pred))
print("\nROC AUC Score:", roc_auc_score(y_test, y_prob))

# Print top 10 most important features
print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Add propensity scores to original dataframe
df['propensity_score'] = model.predict_proba(scaler.transform(X))[:, 1]

# Create donor segments based on propensity scores
df['donor_segment'] = pd.qcut(df['propensity_score'], 
                            q=5, 
                            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

# Display sample results
print("\nSample Donor Segments (Top 10 High Propensity Donors):")
print(df.nlargest(10, 'propensity_score')[['alumni_id', 'propensity_score', 'donor_segment', 
                                          'wealth_score', 'lifetime_giving', 'five_year_donations']])