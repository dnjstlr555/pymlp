# pymlp
```python
import mlp
model=mlp.nsystem(inputSize=2)
model.StackLayer(number=4, function=mlp.ReLU) 
model.StackLayer(number=1, function=mlp.Sigmoid)
x=[[0.03, 0.01], [0.9, 0.97]]
y=[[0], [1]]
for i, data in enumerate(x):
  model.train(inputData=data, result=y[i], trainMethod=mlp.Method.GradientDescent, alpha=0.01)

print(model.fastfeed([0.01, 0.01]))
```

# Model
```python
mlp.nsystem(inputsize=1)
```
returns new nsystem based on given inputsize<br><br>
```python
nsystem.StackLayer(number=3, function=mlp.Normal, defaultSig=1)
```
stack a layer, number meaning layer size. function and defaultSig can be ignored<br><br>
```python
nsystem.fastfeed(input)
```
returns signals of the output layer after input feeded<br><br>
```python
nsystem.train(inputData, result, trainMethod=mlp.Method.GradientDescent, **kwargs)
```
train the model. inputData and result need to have same size of input size and output layer size. training multiple inputData and result data is not allowed <br>
trainMethod can vary. currently supports Gradient Descent and Genetic Algorithm. each method provided kwargs as input.

# Method
## trainning methods
```python
nsystem.train(inputData, result, trainMethod=mlp.Method.GradientDescent, alpha=0.05)
```
alpha meaning learning rate for Gradient Descent train method<br><br>
```python
nsystem.train(inputData, result, trainMethod=mlp.Method.Genetic, poolSize=3, muChance=0.1, muSigma=1, muExp=0, eliteRatio=0.2, randRatio=0.2)
```
poolSize meaning the total size of single generation. mutation will happen by each weights and biases based on muChance(Possibilty). mutation applies using gauss random, muSigma and muExp used sigma and expectation value of gauss random library. some models that have higher fitness will remained in next generation pool, eliteRatio indicates the ratio of those. pure randomly created models will be contained too based on randRatio.
## functions
```python
mlp.Normal=x
mlp.ReLU=sympy.Max(x, 0)
mlp.Sigmoid=1/(1+sympy.exp(x))
```
# Misc
```python
nsystem.save(location='model.pckl')
```
export model as file<br><br>
```python
nsystem.load(location='model.pckl')
```
load model from file<br><br>
```python
nsystem.ToString()
```
returns a string describing the model<br><br>
