

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report, 
    precision_score, 
    recall_score, 
    f1_score
)

# Load Breast Cancer dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Training
        for _ in range(self.n_iterations):
            epoch_loss = 0
            for idx, x_i in enumerate(X):
                # Compute linear prediction
                linear_model = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._activation(linear_model)
                
                # Compute loss and update rule
                loss = y[idx] - y_predicted
                update = self.learning_rate * loss
                
                # Update weights and bias
                self.weights += update * x_i
                self.bias += update
                
                # Accumulate epoch loss
                epoch_loss += loss**2
            
            # Store average epoch loss
            self.losses.append(epoch_loss / n_samples)
        
        return self
    
    def _activation(self, X):
        return np.where(X > 0, 1, 0)
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._activation(linear_model)

# Train Perceptron
perceptron = Perceptron(learning_rate=0.01, n_iterations=1000)
perceptron.fit(X_train, y_train)

# Predict
y_pred = perceptron.predict(X_test)

# Performance Metrics
print("Performance Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Precision, Recall, F1-Score
print("\nDetailed Metrics:")
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))

# Visualization of Learning Curve
plt.figure(figsize=(10, 5))
plt.plot(perceptron.losses)
plt.title('Perceptron Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.tight_layout()
plt.show()

# Feature Importance (weight magnitude)
feature_importance = np.abs(perceptron.weights)
top_features_indices = np.argsort(feature_importance)[-5:]
top_features_names = [cancer.feature_names[i] for i in top_features_indices]

plt.figure(figsize=(10, 5))
plt.bar(top_features_names, feature_importance[top_features_indices])
plt.title('Top 5 Most Important Features')
plt.xlabel('Features')
plt.ylabel('Absolute Weight')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
