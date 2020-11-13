import mlp
import random

sys=mlp.nsystem(inputSize=2)
new=sys.Copy()

print(f"{sys.ToString()} {new.ToString()}")
new.ImportModel(sys)
print(f"{sys.ToString()} {new.ToString()}")