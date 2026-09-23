import sys
import torch
import numpy
import pandas
import sklearn
import transformers

print("=" * 80)
print("PPI Bernett Environment Check")
print("=" * 80)
print("Python       :", sys.version.split()[0])
print("PyTorch      :", torch.__version__)
print("NumPy        :", numpy.__version__)
print("Pandas       :", pandas.__version__)
print("scikit-learn :", sklearn.__version__)
print("Transformers :", transformers.__version__)
print("CUDA         :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU          :", torch.cuda.get_device_name(0))
print("=" * 80)
