import mlp
import random

model=mlp.nsystem(inputSize=2)
model.StackLayer(number=3, function=mlp.ReLU)
model.StackLayer(number=2, function=mlp.Sigmoid)
cnt1=0
cnt2=0
for i in range(500):
    rand = random.choice((0,1))
    
    if rand==0:
        model.train([random.uniform(0.01,0.3),random.uniform(0.01,0.3)], [0.01, 0.01],trainMethod=mlp.Method.GradientDescent,alpha=0.2)
        cnt1+=1
    else:
        model.train([random.uniform(0.8,0.99),random.uniform(0.8,0.99)], [0.99, 0.99], trainMethod=mlp.Method.GradientDescent, alpha=0.2)
        cnt2+=1
    
    if i%10==0: 
        print(f"train for 0 -> {cnt1} / 1 -> {cnt2} / {len(model.memory)}")
        print(f"evaluates for [1,1] {model.evaluate([1,1], [1,1])}")
        print(f"evaluates for [0.1,0.1] {model.evaluate([0.1,0.1], [0, 0])}")


print(model.fastfeed([0.5,0.5]))
print(model.fastfeed([0.1,0.1]))
print(model.fastfeed([0.9,0.9]))
print(model.fastfeed([0,1]))
print(model.fastfeed([33,33]))
print(model.ToString())