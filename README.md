
 # Anomaly Detection Example
 
 This repository contains a minimal example of Gaussian anomaly detection.
 The original notebook has been supplemented with a simple Python module and
 command line entry point so that the core logic can be reused.
 
 ## Requirements
 - Python 3.11+
-- No third-party dependencies are required. Optionally install `matplotlib` if you
-  wish to visualize results.
+- No third-party dependencies are required. Optionally install `matplotlib` if you wish to visualize results.
 
 ## Usage
 Run the demo script which generates a synthetic dataset, fits the Gaussian
 model and prints the selected threshold:
 
 ```bash
 python run_anomaly_detection.py
 ```
 
 The helper functions used by the script live in `utils.py`.
 
 ## Testing
 Unit tests for the helper functions are provided under `tests/` and can be run
 with:
 
 ```bash
 pytest -q
 ```

## Data
Sample training and validation datasets are stored in `data/` and are loaded by `utils.py`.
 
 ## License
 This project is released under the MIT License. See `LICENSE` for details.

