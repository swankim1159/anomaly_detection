+import numpy as np
+import matplotlib.pyplot as plt
+
+DATA_DIR = 'data'
+
+
+def load_data():
+    """Load small 2-feature training and validation datasets."""
+    X_train = np.loadtxt(f"{DATA_DIR}/small_X_train.csv", delimiter=',')
+    X_val = np.loadtxt(f"{DATA_DIR}/small_X_val.csv", delimiter=',')
+    y_val = np.loadtxt(f"{DATA_DIR}/small_y_val.csv", delimiter=',')
+    return X_train, X_val, y_val
+
+
+def load_data_multi():
+    """Load higher-dimensional training and validation datasets."""
+    X_train = np.loadtxt(f"{DATA_DIR}/large_X_train.csv", delimiter=',')
+    X_val = np.loadtxt(f"{DATA_DIR}/large_X_val.csv", delimiter=',')
+    y_val = np.loadtxt(f"{DATA_DIR}/large_y_val.csv", delimiter=',')
+    return X_train, X_val, y_val
+
+
+def multivariate_gaussian(X, mu, var):
+    """Compute the multivariate Gaussian distribution."""
+    k = len(mu)
+    if var.ndim == 1:
+        var = np.diag(var)
+    X = X - mu
+    return (1 / (np.sqrt((2 * np.pi) ** k * np.linalg.det(var))) *
+            np.exp(-0.5 * np.sum(X @ np.linalg.inv(var) * X, axis=1)))
+
+
+def visualize_fit(X, mu, var):
+    """Visualize the dataset and its Gaussian fit."""
+    plt.figure()
+    plt.scatter(X[:, 0], X[:, 1], marker='x', c='b')
+    x1 = np.linspace(min(X[:, 0]), max(X[:, 0]), 60)
+    x2 = np.linspace(min(X[:, 1]), max(X[:, 1]), 60)
+    X1, X2 = np.meshgrid(x1, x2)
+    points = np.c_[X1.ravel(), X2.ravel()]
+    Z = multivariate_gaussian(points, mu, var)
+    Z = Z.reshape(X1.shape)
+    levels = [10 ** h for h in range(-20, 0, 3)]
+    plt.contour(X1, X2, Z, levels=levels, colors='red')
+    plt.xlabel('Latency (ms)')
+    plt.ylabel('Throughput (mb/s)')
+    plt.title('Gaussian Contours')
+    plt.show()
