import random
import numpy as np
import sympy
x,y,z= sympy.symbols('x y z') 

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
    def CreateLayer(self, number=1, inputSize=1):
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
        #ft=sympy.lambdify(x, self.f, np)
        self.sig=self.f.subs(x, self.ProvideZ()) 
    def ProvideZ(self):
        prevSig=self.layer.ProvidePrevSig()
        providz=0
        for i, sig in enumerate(prevSig):
            if self.w[i] is None:
                self.w[i]=random.uniform(0.5,1)
            providz+=self.w[i]*sig
        return providz+self.b
    def deltaHZ(self):
        return sympy.diff(self.f).subs(x,self.ProvideZ())
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
            provid=temp*self.w[i]
            self.w[i]-=temp*n.sig*0.1
            n.back(predict=predict, t=provid)
        self.b-=temp*0.1
    def descback(self, **kwargs):
        if self.layer.index<=0: return 0
        temp=0
        t=None
        predict=None
        str=""
        for key, value in kwargs.items():
            if key=="predict": 
                predict=value
            elif key=="t": 
                t=value
            elif key=="str":
                str=value
        if predict is None: return -1 

        if t is None:
            temp=2/len(self.layer.children)*(self.sig-predict)*(self.deltaHZ()) 
            str+=f"(dCo/dZ{self.layer.index}:{self.index})"
        else:
            temp=t*self.deltaHZ()
            str+=f"(dH{self.layer.index}:{self.index}/dZ{self.layer.index}:{self.index})"
        print(f"{str} => {temp}")
        for i, n in enumerate(self.layer.ProvidePrevNeu()):
            provid=temp*self.w[i]
            sendstr=str
            sendstr+=f"(dZ{self.layer.index}:{self.index}/dH{n.layer.index}:{n.index})"
            self.w[i]-=temp*n.sig*0.1
            n.descback(predict=predict, t=provid, str=sendstr)
        self.b-=temp*0.1
    def ToString(self):
        return f'  ({self.index})sig:{self.sig} weights:{self.w} bias:{self.b}'
    def ToStringOnlySig(self):
        return f'({self.sig})'
Normal=x
ReLU=sympy.Max(0,x)
Sigmoid=1/(1+sympy.exp(x))

def train(sys, inputData, result):
    sys.feed(inputData)
    sys.Activate()
    #print(f'{inputData} -> Precdiction: {result} Predict:{sys.evaluate(result)}')
    for i, n in enumerate(sys.out().children):
        n.back(predict=result[i])
