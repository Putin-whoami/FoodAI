import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for replicability
np.random.seed(42)

# Generate 1000 synthetic samples based on West African (Nigeria) conditions
n_samples = 1000
temperature = np.random.normal(25, 5, n_samples)  # Mean 25°C, std 5°C
humidity = np.random.normal(60, 10, n_samples)    # Mean 60%, std 10%
processing_time = np.random.normal(10, 2, n_samples)  # Mean 10 hours, std 2 hours
power_downtime = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # 20% downtime

# CORRECTED: Define baseline waste rate (25% FAO 2023) with more realistic thresholds
baseline_waste_rate = 0.25

# more realistic waste prediction model
#using stricter thresholds and weighted probability instead of simple OR logic
def calculate_waste_probability(temp, hum, proc_time, downtime):
    """Calculate waste probability based on multiple factors"""
    prob = 0.0
    
    # Temperature factor (waste increases significantly above 30°C)
    if temp > 30:
        prob += 0.4 * ((temp - 30) / 10)
    elif temp > 28:
        prob += 0.2 * ((temp - 28) / 2)
    
    # Humidity factor (waste increases above 70%)
    if hum > 70:
        prob += 0.3 * ((hum - 70) / 20)
    elif hum > 65:
        prob += 0.15 * ((hum - 65) / 5)
    
    # Processing time factor (waste increases above 12 hours)
    if proc_time > 12:
        prob += 0.25 * ((proc_time - 12) / 4)
    elif proc_time > 11:
        prob += 0.1 * ((proc_time - 11) / 1)
    
    # Power downtime (major factor)
    if downtime == 1:
        prob += 0.6
    
    # Base waste probability
    prob += 0.05
    
    return min(prob, 1.0)  # Cap at 100%

# Calculate waste probabilities for each sample
waste_probabilities = [calculate_waste_probability(t, h, p, d) 
                      for t, h, p, d in zip(temperature, humidity, processing_time, power_downtime)]

# Generate actual waste based on probabilities (this represents the baseline scenario)
baseline_waste = np.random.binomial(1, waste_probabilities)

# Simulate AI intervention effect
# AI can predict high-risk scenarios and intervene to reduce waste by 60%
ai_success_rate = 0.6
high_risk_threshold = 0.3

# Identify high-risk scenarios
high_risk_scenarios = np.array(waste_probabilities) > high_risk_threshold

# AI intervention reduces waste in 60% of high-risk cases
ai_intervention_success = np.random.binomial(1, ai_success_rate, n_samples)
ai_prevented_waste = high_risk_scenarios & (baseline_waste == 1) & (ai_intervention_success == 1)

# Final waste after AI intervention
waste_with_ai = baseline_waste & ~ai_prevented_waste

# Calculate actual waste rates
actual_baseline_rate = np.mean(baseline_waste)
actual_ai_rate = np.mean(waste_with_ai)
actual_waste_reduction = (actual_baseline_rate - actual_ai_rate) * 100

print(f"Actual Baseline Waste Rate: {actual_baseline_rate:.2%}")
print(f"Waste Rate with AI: {actual_ai_rate:.2%}")
print(f"Waste Reduction Achieved: {actual_waste_reduction:.1f} percentage points")

# Create DataFrame for analysis
data = pd.DataFrame({
    'Temperature (°C)': temperature,
    'Humidity (%)': humidity,
    'Processing Time (hours)': processing_time,
    'Power Downtime': power_downtime,
    'Waste Probability': waste_probabilities,
    'Baseline Waste': baseline_waste,
    'Waste with AI': waste_with_ai
})

# Prepare features and target for model training
X = data[['Temperature (°C)', 'Humidity (%)', 'Processing Time (hours)', 'Power Downtime']]
y = data['Waste with AI']  # Train on AI-improved outcomes

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Model Accuracy: {accuracy:.2%}")
print(f"Model predicts waste rate of: {np.mean(y_pred):.2%}")

# Enhanced Visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Enhanced Confusion Matrix
plt.subplot(2, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Count'},
            xticklabels=['No Waste', 'Waste'], yticklabels=['No Waste', 'Waste'])
plt.title('Waste Prediction Performance\n(AI-Enhanced Framework)', fontsize=12, pad=15)
plt.xlabel('Predicted Label', fontsize=10)
plt.ylabel('Actual Label', fontsize=10)

# 2. Corrected Waste Reduction Bar Chart
plt.subplot(2, 2, 2)
waste_rates = [actual_baseline_rate * 100, actual_ai_rate * 100]
labels = [f'Baseline Waste\n({actual_baseline_rate:.1%})', f'AI-Enhanced\n({actual_ai_rate:.1%})']
colors = ['#FF6B6B', '#4ECDC4']
bars = plt.bar(labels, waste_rates, color=colors, edgecolor='black', linewidth=1.5)
plt.title('Waste Reduction Impact of AI Framework\n(Nigeria Food Industry)', fontsize=12, pad=15)
plt.ylabel('Waste Rate (%)', fontsize=10)
plt.ylim(0, max(waste_rates) * 1.2)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 3. Scatter Plot for Input Parameters vs. Waste (Baseline)
plt.subplot(2, 2, 3)
scatter = plt.scatter(data['Temperature (°C)'], data['Humidity (%)'], 
                     c=data['Baseline Waste'], cmap='RdYlGn_r',
                     s=60, alpha=0.6, edgecolors='w', linewidth=0.5)
plt.colorbar(scatter, label='Baseline Waste (0 = No, 1 = Yes)', ticks=[0, 1])
plt.title('Temperature vs. Humidity\n(Baseline Waste Pattern)', fontsize=12, pad=15)
plt.xlabel('Temperature (°C)', fontsize=10)
plt.ylabel('Humidity (%)', fontsize=10)
plt.axvline(x=28, color='orange', linestyle='--', alpha=0.7, label='Warning')
plt.axvline(x=30, color='red', linestyle='--', alpha=0.7, label='Critical')
plt.axhline(y=65, color='orange', linestyle='--', alpha=0.7)
plt.axhline(y=70, color='red', linestyle='--', alpha=0.7)

# 4. Scatter Plot for AI-Enhanced Results
plt.subplot(2, 2, 4)
scatter2 = plt.scatter(data['Temperature (°C)'], data['Humidity (%)'], 
                      c=data['Waste with AI'], cmap='RdYlGn_r',
                      s=60, alpha=0.6, edgecolors='w', linewidth=0.5)
plt.colorbar(scatter2, label='AI-Enhanced Waste (0 = No, 1 = Yes)', ticks=[0, 1])
plt.title('Temperature vs. Humidity\n(AI-Enhanced Results)', fontsize=12, pad=15)
plt.xlabel('Temperature (°C)', fontsize=10)
plt.ylabel('Humidity (%)', fontsize=10)
plt.axvline(x=28, color='orange', linestyle='--', alpha=0.7)
plt.axvline(x=30, color='red', linestyle='--', alpha=0.7)
plt.axhline(y=65, color='orange', linestyle='--', alpha=0.7)
plt.axhline(y=70, color='red', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Additional Analysis: Waste Reduction by Risk Category
print(f"\n--- Detailed Analysis ---")
low_risk = data[data['Waste Probability'] <= 0.2]
medium_risk = data[(data['Waste Probability'] > 0.2) & (data['Waste Probability'] <= 0.5)]
high_risk = data[data['Waste Probability'] > 0.5]

print(f"Low Risk Scenarios ({len(low_risk)} samples):")
print(f"  Baseline waste rate: {np.mean(low_risk['Baseline Waste']):.2%}")
print(f"  AI-enhanced rate: {np.mean(low_risk['Waste with AI']):.2%}")

print(f"Medium Risk Scenarios ({len(medium_risk)} samples):")
print(f"  Baseline waste rate: {np.mean(medium_risk['Baseline Waste']):.2%}")
print(f"  AI-enhanced rate: {np.mean(medium_risk['Waste with AI']):.2%}")

print(f"High Risk Scenarios ({len(high_risk)} samples):")
print(f"  Baseline waste rate: {np.mean(high_risk['Baseline Waste']):.2%}")
print(f"  AI-enhanced rate: {np.mean(high_risk['Waste with AI']):.2%}")

# Economic Impact Estimation
print(f"\n--- Economic Impact Estimation ---")
avg_batch_value = 50000  # Average value per batch in Naira
batches_per_year = 365
annual_waste_reduction = actual_waste_reduction / 100 * batches_per_year * avg_batch_value
print(f"Estimated annual savings: ₦{annual_waste_reduction:,.0f}")
print(f"Estimated monthly savings: ₦{annual_waste_reduction/12:,.0f}")