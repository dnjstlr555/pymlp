import mlp
import random

model=mlp.nsystem(inputSize=2)
model.StackLayer(number=3, function=mlp.ReLU)
model.StackLayer(number=1, function=mlp.Sigmoid)


for i in range(500):
    rand = random.choice((0,1))
    if rand==0:
        model.train([random.uniform(0.7,1),random.uniform(0.7,1)], [1], trainMethod=mlp.Method.Genetic, poolSize=10, muChance=0.05, eliteRatio=0.4, randRatio=0.05)
    else:
        model.train([random.uniform(0,0.3),random.uniform(0,0.3)], [0], trainMethod=mlp.Method.Genetic, poolSize=10, muChance=0.05, eliteRatio=0.2, randRatio=0.05)
    if i%10==0:
        print(f"{model.evaluate([0,0], [0])} {model.evaluate([0.99,0.99], [1])}")

print(f"{model.fastfeed([0,0])} {model.fastfeed([1,1])}")

new=model.Copy()
new.__reset__()

for i in range(500):
    rand = random.choice((0,1))
    if rand==0:
        new.train([random.uniform(0.7,1),random.uniform(0.7,1)], [1], trainMethod=mlp.Method.GradientDescent, alpha=0.1)
    else:
        new.train([random.uniform(0,0.3),random.uniform(0,0.3)], [0], trainMethod=mlp.Method.GradientDescent, alpha=0.1)
    if i%10==0:
        print(f"{new.evaluate([0,0], [0])} {new.evaluate([0.99,0.99], [1])}")

print(f"{new.fastfeed([0,0])} {new.fastfeed([1,1])}")