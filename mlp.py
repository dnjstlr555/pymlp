import random
from sympy import *
from sympy.abc import A,B
x= Symbol('x')

class nsystem:
    def __init__(self):
        self.children = []
    def Add(self):
        newLayer=nlayer(self)
        self.children.append(newLayer)
        newLayer.index=self.children.index(newLayer)
        return newLayer
    def ProvidePrevLayer(self, layer):
        if layer.index>0:
            return self.children[layer.index-1]
        else:
            return layer
    def feed(self, input):
        for i, n in enumerate(self.children[0].children):
            if input[i] is not None:
                n.sig=input[i]
            else:
                print("Input size does not match")
                return
    def out(self):
        return self.children[len(self.children)-1]
    def Activate(self):
        for l in self.children:
            l.ActivateLayer()
    def ToString(self):
        nstr=f'Sys -->\n'
        for n in self.children:
            nstr+=f'{n.ToString()}\n'
        return nstr
    def ToStringOnlySig(self):
        nstr=f'Sys -->'
        for n in self.children:
            nstr+=f'{n.ToStringOnlySig()} '
        return nstr

class nlayer:
    def __init__(self, sys=None):
        self.children = []
        self.index = None
        self.sys = sys
    def ToString(self):
        nstr=f'Layer {self.index} -->\n'
        for n in self.children:
            nstr+=f'{n.ToString()}\n'
        return nstr
    def ToStringOnlySig(self):
        nstr=f'\n Layer {self.index} -->\n'
        for n in self.children:
            nstr+=f'{n.ToStringOnlySig()} '
        return nstr
    def Add(self, f):
        newNeu=nneuron(self)
        self.children.append(newNeu)
        newNeu.index=self.children.index(newNeu)
        newNeu.f=f
        return newNeu
    def GetAllSig(self):
        sigs=[]
        for n in self.children:
            sigs.append(n.sig)
        return sigs
    def ProvidePrevSig(self):
        prevLayer=self.sys.ProvidePrevLayer(self)
        prevSig=prevLayer.GetAllSig()
        return prevSig
    def ActivateLayer(self):
        if self.index is 0:
            return
        for n in self.children:
            n.ActivateEach()
    
class nneuron:
    def __init__(self, layer=None):
        self.layer = layer
        self.index = None
        self.sig = 0
        self.w = [None for x in range(len(self.layer.ProvidePrevSig()))]
        self.b = 0
        self.f = Normal
        self.zi = (x*A)
        self.z=lambdify([x, A], self.zi)
    def ActivateEach(self):
        prevSig=self.layer.ProvidePrevSig()
        self.sig=0
        f=lambdify(x, self.f)
        for i, sig in enumerate(prevSig):
            if self.w[i] is None:
                self.w[i]=random.uniform(-1,1)
            self.sig+=self.z(sig,self.w[i])
        self.sig=f(self.sig+self.b)
    def ProvideZ(self):
        prevSig=self.layer.ProvidePrevSig()
        providz=0
        for i, sig in enumerate(prevSig):
            if self.w[i] is None:
                self.w[i]=random.uniform(-1,1)
            providz+=self.z(sig,self.w[i])
        return providz+self.b
    def ToString(self):
        return f'index:{self.index}\nsig:{self.sig}\nweights:{self.w}\nbias:{self.b}'
    def ToStringOnlySig(self):
        return f'({self.sig})'

def CreateLayer(number,sys, inputSize):
    layer=[]
    for i in range(number):
        layer.append(sys.Add())
    CreateNeurons(inputSize, layer[0], Normal, 0)
    return layer

def CreateNeurons(number, layer, function, sig):
    neu=[]
    for i in range(number):
        n=layer.Add(function)
        n.sig=sig
        neu.append(n)
    return neu

def train(sys, inputData, result):
    sys.feed(inputData)
    sys.Activate()
    out = sys.out()
    score=0
    for layer in sys.children[::-1]:
        for i, neu in enumerate(layer.children):
            if neu.layer.index<=0:
                break
            dc=diff(Cost, x)
            da=diff(neu.f,x)
            t=dc.subs(x, neu.sig).subs(A, result[i])*da.subs(x, neu.ProvideZ())
            print(neu.ToString())
            for i, weight in enumerate(neu.w):
                neu.w[i]=weight - (t*neu.layer.ProvidePrevSig()[i])
            neu.b=neu.b - t*1
            print(t)
            print(neu.ToString())

ReLU=Max(0,x)
Normal=x
Cost=(x-A)**2
c=lambdify([x, A], Cost)

sys=nsystem()
layer=CreateLayer(3, sys, 2)
CreateNeurons(2, layer[1], ReLU, 1)
CreateNeurons(2, layer[2], ReLU, 1)


print(sys.ToStringOnlySig())
train(sys, [1,0], [1,1])
train(sys, [1,0], [1,1])
train(sys, [1,0], [1,1])
train(sys, [1,0], [1,1])
print(sys.ToStringOnlySig())
