from pysurvival.models.semi_parametric import CoxPHModel
from pysurvival.models.multi_task import LinearMultiTaskModel
from pysurvival.datasets import Dataset
from pysurvival.utils.metrics import concordance_index
import numpy as np
import torch
import sys # for python version

# For reproducibility of results
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
else:
    torch.manual_seed(0)

print(f"Running smoke test with Python {sys.version}")
print(f"PyTorch version: {torch.__version__}")

# Loading and splitting a simple example into train/test sets
print("Loading dataset...")
try:
    X_train, T_train, E_train, X_test, T_test, E_test =         Dataset('simple_example').load_train_test()
    print("Dataset loaded successfully.")
    print(f"X_train shape: {X_train.shape}, T_train shape: {T_train.shape}, E_train shape: {E_train.shape}")
    print(f"X_test shape: {X_test.shape}, T_test shape: {T_test.shape}, E_test shape: {E_test.shape}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Building a CoxPH model
print("Fitting CoxPHModel...")
try:
    coxph_model = CoxPHModel()
    coxph_model.fit(X=X_train, T=T_train, E=E_train, init_method='he_uniform',
                    l2_reg = 1e-4, lr = .4, tol = 1e-4, verbose=False)
    print("CoxPHModel fitted successfully.")
except Exception as e:
    print(f"Error fitting CoxPHModel: {e}")
    exit(1)

# Building a MTLR model
print("Fitting LinearMultiTaskModel (MTLR)...")
try:
    mtlr_model = LinearMultiTaskModel()
    mtlr_model.fit(X=X_train, T=T_train, E=E_train, init_method = 'glorot_uniform',
               optimizer ='adam', lr = 8e-4, verbose=False)
    print("LinearMultiTaskModel fitted successfully.")
except Exception as e:
    print(f"Error fitting LinearMultiTaskModel: {e}")
    exit(1)

# Checking the model performance
print("Calculating concordance index for CoxPHModel...")
try:
    c_index1 = concordance_index(model=coxph_model, X=X_test, T=T_test, E=E_test )
    print(f"CoxPHModel c-index = {c_index1:.2f}")
except Exception as e:
    print(f"Error calculating C-index for CoxPHModel: {e}")
    exit(1)

print("Calculating concordance index for LinearMultiTaskModel...")
try:
    c_index2 = concordance_index(model=mtlr_model, X=X_test, T=T_test, E=E_test )
    print(f"LinearMultiTaskModel c-index = {c_index2:.2f}")
except Exception as e:
    print(f"Error calculating C-index for LinearMultiTaskModel: {e}")
    exit(1)

print("Smoke test completed successfully.")
