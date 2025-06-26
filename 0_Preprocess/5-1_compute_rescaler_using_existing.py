import os 
import nibabel  # if needed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor

# -------------------------
# Read data from file
# -------------------------
data = "./ratio.txt"
with open(data, 'r') as f:
    datalines = f.readlines()

height = []
distance = []
for line in datalines:
    h, d, r = line.split(sep=' ')
    height.append(float(h))
    distance.append(float(d))

# Convert lists to numpy arrays
X_all = np.array(height).reshape(-1, 1)  # features must be 2D
y_all = np.array(distance)

# -------------------------
# Preliminary Fit to Estimate Noise Level
# -------------------------
prelim_model = LinearRegression()
prelim_model.fit(X_all, y_all)
y_pred_prelim = prelim_model.predict(X_all)
residuals = np.abs(y_all - y_pred_prelim)

# Compute the median absolute deviation (MAD)
mad = np.median(np.abs(residuals - np.median(residuals)))
sigma_estimated = 1.4826 * mad

# -------------------------
# Determine the Residual Threshold
# -------------------------
k = 2  # Multiplier for sigma, can be tuned
residual_threshold = k * sigma_estimated

print(f"Estimated sigma: {sigma_estimated:.2f}")
print(f"Residual threshold used in RANSAC: {residual_threshold:.2f}")

# -------------------------
# Fit model using RANSAC
# -------------------------
ransac = RANSACRegressor(
    estimator=LinearRegression(),  # use 'estimator' for newer scikit-learn versions
    residual_threshold=residual_threshold,
    random_state=42
)
ransac.fit(X_all, y_all)

# Extract inlier mask (optional visualization)
inlier_mask = ransac.inlier_mask_
outlier_mask = ~inlier_mask

# -------------------------
# Extract and Print the Regression Equation
# -------------------------
slope = ransac.estimator_.coef_[0]
intercept = ransac.estimator_.intercept_
print(f"Regression equation: y = {slope:.3f} * x + {intercept:.3f}")
r2_inliers = ransac.score(X_all[inlier_mask], y_all[inlier_mask])
print(f"R^2 on inlier data: {r2_inliers:.3f}")
# -------------------------
# Visualize the results
# -------------------------
line_X = np.linspace(X_all.min(), X_all.max(), 100).reshape(-1, 1)
line_y_ransac = ransac.predict(line_X)

plt.figure(figsize=(10, 6))
plt.scatter(X_all[inlier_mask], y_all[inlier_mask], color='blue', marker='o', label='Inliers')
plt.scatter(X_all[outlier_mask], y_all[outlier_mask], color='red', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='green', linewidth=2, label='RANSAC Regression')
plt.xlabel("Height")
plt.ylabel("Distance")
plt.title("Linear Regression using RANSAC (Inliers Only)")
plt.legend(loc='upper left')
plt.show()
