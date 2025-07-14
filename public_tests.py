+import numpy as np
+
+def select_threshold_test(func):
+    """Basic sanity check for select_threshold implementation."""
+    y_val = np.array([1, 0, 1, 0])
+    p_val = np.array([0.1, 0.2, 0.05, 0.9])
+    eps, f1 = func(y_val, p_val)
+    assert isinstance(eps, float)
+    assert isinstance(f1, float)
+    print("select_threshold passed basic test")
+
