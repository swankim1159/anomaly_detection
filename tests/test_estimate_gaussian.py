+from anomaly import estimate_gaussian
+
+
+def test_estimate_gaussian():
+    X = [
+        [1.0, 2.0],
+        [2.0, 1.0],
+        [3.0, 0.0],
+    ]
+    mu, var = estimate_gaussian(X)
+    assert mu == [2.0, 1.0]
+    assert var == [2/3, 2/3]
