import random
from sympy import *
from sympy.abc import A,B
import numpy as np
x,y,z= symbols('x y z')

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
    def fastfeed(self, input):
        self.feed(input)
        self.Activate()
        return self.out().GetAllSig()
    def evaluate(self, result):
        temp=0
        for i, n in enumerate(self.out().children):
            temp+=(n.sig-result[i])**2
        return temp/len(self.out().children)
    def Activate(self):
        for l in self.children:
            l.ActivateLayer()
    def CreateLayer(self, number, inputSize):
        layer=[]
        for i in range(number):
            layer.append(self.Add())
        self.CreateNeurons(inputSize, layer[0], Normal, 0)
        return layer
    def CreateNeurons(self, number, layer, function, sig):
        neu=[]
        for i in range(number):
            n=layer.Add(function)
            n.sig=sig
            neu.append(n)
        return neu
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
    def ProvidePrevNeu(self):
        prevLayer=self.sys.ProvidePrevLayer(self)
        return prevLayer.children
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
    def ActivateEach(self):
        self.sig=0
        f=lambdify(x, self.f)
        self.sig=f(self.ProvideZ())
    def ProvideZ(self):
        prevSig=self.layer.ProvidePrevSig()
        providz=0
        for i, sig in enumerate(prevSig):
            if self.w[i] is None:
                self.w[i]=1 #random.uniform(,1)
            providz+=self.w[i]*sig
        return providz+self.b
    def deltaHZ(self):
        return diff(self.f).subs(x,self.ProvideZ())
    def back(self, **kwargs):
        if self.layer.index<=0: return 0
        temp=0
        t=None
        predict=[]
        for key, value in kwargs.items():
            if key=="predict": 
                predict=value
            elif key=="t": 
                t=value
        if predict is None: return -1      
        if t is None:
            temp=2/len(self.layer.children)*(self.sig-predict)*(self.deltaHZ())
        else:
            temp=t*self.deltaHZ()  
        for i, n in enumerate(self.layer.ProvidePrevNeu()):
            #print(f"NEW W{self.layer.index}_{self.index}/(prev{i}) {self.w[i]}->{self.w[i]-(temp*n.sig)} temp:{temp} t:{t} n.sig:{n.sig} (self.deltaHZ:{self.deltaHZ()} self.sig:{self.sig} predict:{predict} len : {2/len(self.layer.children)}")
            provid=temp*self.w[i]
            self.w[i]-=temp*n.sig*0.1
            n.back(predict=predict, t=provid)
        self.b-=temp*0.1
    def ToString(self):
        return f'  ({self.index})sig:{self.sig} weights:{self.w} bias:{self.b}'
    def ToStringOnlySig(self):
        return f'({self.sig})'
Normal=x
ReLU=Max(0,x)


def train(sys, inputData, result):
    sys.feed(inputData)
    sys.Activate()
    #print(f'{inputData} -> Precdiction: {result} Predict:{sys.evaluate(result)}')
    for i, n in enumerate(sys.out().children):
        n.back(predict=result[i])
    

sys=nsystem()
layer=sys.CreateLayer(3, 2)
sys.CreateNeurons(2, layer[1], ReLU, 1)
sys.CreateNeurons(1, layer[2], ReLU, 1)


for i in range(500):
    rand = random.choice((0,1))
    if rand==0:
        train(sys, [random.uniform(0,0.3),random.uniform(0,0.3)], [0])
    else:
        train(sys, [random.uniform(0.7,1),random.uniform(0.7,1)], [1])

print(sys.ToString())
print(sys.fastfeed([0.5,0.5]))
print(sys.fastfeed([0.1,0.1]))
print(sys.fastfeed([1,1]))
