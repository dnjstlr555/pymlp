# pymlp
Multi Layer Perceptron system for python made for study.<br>
Every layers can connect(feeded) directly to other layers, thus you can freely construct deep neural network models.<br>
Check out the test.py file to see example of this. Docs file is outdated (Works for mlp_old.py)

# mlp_old.py
Fully Connected layer implementation for study purpose. <br>supports creation of MLP system `model=mlp.nsystem(inputsize=n)` and adding layer to system by `model.add(number=3, function=mlp.ReLU, defaultSig=1)` and train by `model.train(x, y, trainMethod=mlp.method.GradientDescent, series=True, verbose=0, **kwargs)`. <br>for trainning model, supports train methods for SGD, [Genetic Algorithm](https://www.mdpi.com/1099-4300/22/11/1239/pdf). for more information, visit [docs/docs.md](https://github.com/dnjstlr555/pymlp/blob/main/docs/docs.md). Implementation and study note in korean can be found in [docs/createlog.md](https://github.com/dnjstlr555/pymlp/blob/main/docs/createlog.md).<br>

# Installation
for both version, requires numpy to run. type `pip install numpy` on terminal.
