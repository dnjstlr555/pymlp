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
nsystem.train(inputData, result, trainMethod=mlp.method.GradientDescent, series=False, verbose=0, **kwargs)
```
train the model. inputData and result need to have same size of input size and output layer size. training multiple inputData and result data is not allowed by default. make sure the series attribute set to be True when you provide continuous inputData(two dimensional variable) and result(also two dimensional variable). each data needs to be correspond and have same size as inputSize and output layer number.when verbose is set to 1 and the data is series, train result will be printed every 10 times while training. otherwise print after train is finished<br>
inputData and result must be an array.<br>
trainMethod can vary. currently supports Gradient Descent and Genetic Algorithm. each method provided kwargs as input.
# Method
## trainning methods
```python
nsystem.train(inputData, result, trainMethod=mlp.method.GradientDescent, alpha=0.05)
```
alpha meaning learning rate for Gradient Descent train method<br><br>
```python
nsystem.train(inputData, result, trainMethod=mlp.method.Genetic, poolSize=3, muChance=0.1, muSigma=1, muExp=0, eliteRatio=0.2, randRatio=0.2)
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
```python
nsystem.evaluate(inputData, result)
```
returns diffrence of output of current model between expected result. those two must be an array<br><br>

# Advance
```python
nsystem.Add()
```
returns empty layer added into model<br><br>
```python
(w,b)=nsystem.GetParams()
```
returns w,b table<br><br>
```python
nsystem.SetParams(w, b)
```
set params.<br><br>
```python
nsystem.Proto()
```
returns exact copy of model<br><br>
```python
nsystem.ImportProto(proto)
```
import model from proto<br><br>
```python
nsystem.ImportModel(model)
```
import model directly from other model<br><br>
```python
nlayer.Add(f=mlp.Normal)
```
returns added neuron<br><br>
