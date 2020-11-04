import mlp
import random

sys=mlp.nsystem()
layer=sys.CreateLayer(4,2)
sys.CreateNeurons(4, layer[1], mlp.ReLU, 1)
sys.CreateNeurons(4, layer[2], mlp.ReLU, 1)
sys.CreateNeurons(2, layer[3], mlp.Sigmoid, 1)

sys.feed([0.3,0.3])
sys.Activate()
result=[0.9,0.9]
for i, n in enumerate(sys.out().children):
    n.descback(predict=result[i])