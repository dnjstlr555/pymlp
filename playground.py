import mlp
import random

model=mlp.nsystem(inputSize=2)
model.StackLayer(number=3, function=mlp.ReLU)
model.StackLayer(number=1, function=mlp.Sigmoid)

x,y=([],[])
for i in range(500):
    rand = random.choice((0,1))
    if rand==0:
        x.append([random.uniform(0.7,1),random.uniform(0.7,1)])
        y.append([1])
    else:
        x.append([random.uniform(0,0.3),random.uniform(0,0.3)])
        y.append([0])
model.train(x, y, trainMethod=mlp.method.Genetic, poolSize=3, muChance=0.05, eliteRatio=0.2, randRatio=0.05, series=True, verbose=1)
model.train(x, y, trainMethod=mlp.method.GradientDescent, alpha=0.05, series=True, verbose=1)


print(f"{model.fastfeed([0,0])} {model.fastfeed([1,1])}")