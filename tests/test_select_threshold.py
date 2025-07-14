+from anomaly import select_threshold
+
+
+def test_select_threshold():
+    p_val = [
+        5e-05, 4e-05, 3e-05, 6e-05, 7e-05, 2e-05, 8.95e-05, 1e-04,
+        7e-05, 0.0001, 0.00015, 0.00012, 0.00011, 0.00016, 0.0002, 0.00009
+    ]
+    y_val = [1]*8 + [0]*8
+    epsilon, F1 = select_threshold(y_val, p_val)
+    assert abs(epsilon - 8.966e-05) < 1e-09
+    assert abs(F1 - 0.875) < 1e-09
