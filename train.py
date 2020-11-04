import mlp
import random

sys=mlp.nsystem()
layer=sys.CreateLayer(3,2)
sys.CreateNeurons(4, layer[1], mlp.ReLU, 1)
sys.CreateNeurons(2, layer[2], mlp.Sigmoid, 1)

cnt1=0
cnt2=0
for i in range(1000):
    rand = random.choice((0,1))
    
    if rand==0:
        mlp.train(sys, [random.uniform(0.01,0.3),random.uniform(0.01,0.3)], [0.01, 0.01])
        cnt1+=1
    else:
        mlp.train(sys, [random.uniform(0.8,0.99),random.uniform(0.8,0.99)], [0.99, 0.99])
        cnt2+=1
    
    if i%10==0: 
        sys.fastfeed([1,1])
        print(f"train for 0 -> {cnt1} / 1 -> {cnt2}")
        print(f"evaluates for [1,1] {sys.evaluate([1, 1])}")
        sys.fastfeed([0.1,0.1])
        print(f"evaluates for [0.1,0.1] {sys.evaluate([0, 0])}")

print(sys.fastfeed([0.5,0.5]))
print(sys.fastfeed([0.1,0.1]))
print(sys.fastfeed([0.9,0.9]))
print(sys.fastfeed([0,1]))
print(sys.fastfeed([33,33]))
print(sys.ToString())