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

# Define baseline waste rate (25% FAO 2023)
baseline_waste_rate = 0.25
waste = (temperature > 25) | (humidity > 60) | (processing_time > 10) | (power_downtime == 1)
waste = np.where(waste, 1, 0)  # 1 = waste, 0 = no waste

# DataFrame for better organization
data = pd.DataFrame({
    'Temperature (°C)': temperature,
    'Humidity (%)': humidity,
    'Processing Time (hours)': processing_time,
    'Power Downtime': power_downtime,
    'Waste': waste
})

# Split data into training and testing sets
X = data[['Temperature (°C)', 'Humidity (%)', 'Processing Time (hours)', 'Power Downtime']]
y = data['Waste']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model to predict waste
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
predicted_waste_rate = np.mean(y_pred)
waste_reduction = (baseline_waste_rate - predicted_waste_rate) * 100

# Print results
print(f"Model Accuracy: {accuracy:.2f}")
print(f"Predicted Waste Rate: {predicted_waste_rate:.2f} (Baseline: 25%)")
print(f"Waste Reduction: {waste_reduction:.2f}%")

# Enhanced Visualizations
sns.set_style("whitegrid")  # Use Seaborn's whitegrid style

# 1. Enhanced Confusion Matrix
plt.figure(figsize=(10, 7))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Count'},
            xticklabels=['No Waste', 'Waste'], yticklabels=['No Waste', 'Waste'])
plt.title('Waste Prediction Performance\n(Modular AI Framework)', fontsize=14, pad=20)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('Actual Label', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j + 0.5, i + 0.5, cm[i, j], ha='center', va='center', color='black', fontsize=20)
plt.tight_layout()
plt.show()

# 2. Enhanced Waste Reduction Bar Chart
plt.figure(figsize=(10, 6))
waste_rates = [baseline_waste_rate * 100, predicted_waste_rate * 100]
labels = ['Baseline Waste (25%)', 'Predicted Waste with AI']
colors = ['#FF6B6B', '#Seaborn’s4ECDC4']  # Appealing colors: red, teal
bars = plt.bar(labels, waste_rates, color=colors, edgecolor='black', linewidth=1.5)
plt.title('Waste Reduction Impact of AI Framework\n(Nigeria Food Industry)', fontsize=14, pad=15)
plt.ylabel('Waste Rate (%)', fontsize=12)
plt.ylim(0, 30)  # Set y-axis limit for clarity
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# 3. Scatter Plot for Input Parameters vs. Waste
plt.figure(figsize=(10, 6))
scatter = plt.scatter(data['Temperature (°C)'], data['Humidity (%)'], c=data['Waste'], cmap='RdYlGn',
                      s=100, alpha=0.6, edgecolors='w', linewidth=0.5)
plt.colorbar(scatter, label='Waste (0 = No, 1 = Yes)', ticks=[0, 1])
plt.title('Temperature vs. Humidity Impact on Waste\n(Nigeria Food Processing)', fontsize=14, pad=15)
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Humidity (%)', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.axvline(x=25, color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=60, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
