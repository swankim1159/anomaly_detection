+def estimate_gaussian(X):
+    """Calculate mean and variance for each feature in X.
+
+    Parameters
+    ----------
+    X : list of list of float
+        Dataset with m rows and n columns.
+
+    Returns
+    -------
+    tuple of two lists (mu, var)
+        mu: list of means of each feature
+        var: list of variances of each feature
+    """
+    if not X:
+        return [], []
+    m = len(X)
+    n = len(X[0])
+    mu = [0.0 for _ in range(n)]
+    for row in X:
+        for j, val in enumerate(row):
+            mu[j] += val
+    mu = [v / m for v in mu]
+    var = [0.0 for _ in range(n)]
+    for row in X:
+        for j, val in enumerate(row):
+            diff = val - mu[j]
+            var[j] += diff * diff
+    var = [v / m for v in var]
+    return mu, var
+
+def select_threshold(y_val, p_val):
+    """Find the best threshold epsilon using F1 score."""
+    if not y_val or not p_val:
+        return 0.0, 0.0
+    min_p, max_p = min(p_val), max(p_val)
+    step_size = (max_p - min_p) / 1000
+    best_epsilon = min_p
+    best_F1 = -1.0
+    epsilon = min_p
+    while epsilon < max_p:
+        predictions = [pv < epsilon for pv in p_val]
+        tp = sum(pred and y for pred, y in zip(predictions, y_val))
+        fp = sum(pred and not y for pred, y in zip(predictions, y_val))
+        fn = sum((not pred) and y for pred, y in zip(predictions, y_val))
+        if tp + fp == 0 or tp + fn == 0:
+            F1 = 0.0
+        else:
+            prec = tp / (tp + fp)
+            rec = tp / (tp + fn)
+            if prec + rec == 0:
+                F1 = 0.0
+            else:
+                F1 = 2 * prec * rec / (prec + rec)
+        if F1 > best_F1:
+            best_F1 = F1
+            best_epsilon = epsilon
+        epsilon += step_size
+    return best_epsilon, best_F1
