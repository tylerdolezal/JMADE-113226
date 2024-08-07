import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Generate random X and y matrices for demonstration
Xscaler = StandardScaler()
yscaler = StandardScaler()

X = Xscaler.fit_transform(np.load("x-matrix.npy"))
y = yscaler.fit_transform(np.load("y-matrix.npy"))
# Create a Random Forest Regressor model
rf_regressor = RandomForestRegressor(n_estimators=50, random_state=42)

# Create a k-fold cross-validator
kf = KFold(n_splits=10, shuffle=True, random_state=42)

mse_scores = []  # List to store mean squared error scores

# Perform k-fold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the model
    rf_regressor.fit(X_train, y_train)


    # Make predictions on the test set
    y_pred = rf_regressor.predict(X_test)

    # Calculate mean squared error and append it to the scores list
    y_test = yscaler.inverse_transform(y_test)
    y_pred = yscaler.inverse_transform(y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(np.sqrt(mse))

# Calculate and print the mean and standard deviation of the mean squared error scores
mse_mean = np.mean(mse_scores)
mse_std = np.std(mse_scores)
print(f"Root Mean Squared Error (mean): {mse_mean} GPa")
print(f"Root Mean Squared Error (std): {mse_std} GPa")


# Get feature importances
feature_importances = rf_regressor.feature_importances_

import matplotlib.pyplot as plt
print(X.shape)
# Plot feature importances
#chunks = [(0, 9), (9, 9+48), (9+48, 9+48+144)]
chunks = [(0,288), (288,X.shape[1])]
#chunks = [(0,9), (9,X.shape[1])]
fig,ax = plt.subplots()
pie_values = []

for start, end in chunks:
    chunk_values = feature_importances[start:end]
    pie_values.append(sum(chunk_values))

labels = ["Encoded Positions", "Lattice Vectors"]
#labels = ["Lattice Vectors", "Encoded Positions"]
plt.pie(pie_values, labels = labels, autopct='%.1f%%')

plt.show()

# save the transformation used for model training
from joblib import dump

dump(rf_regressor, 'rf_model_for_constants.pkl')
dump(Xscaler,"rf_Xscaler.joblib")
dump(yscaler, "rf_yscaler.joblib")
