import random
import numpy as np
import math
import sympy
import itertools
from copy import deepcopy
x,y,z= sympy.symbols('x y z') 

class nsystem:
    def __init__(self, inputSize=1):
        self.children = []
        self.memory = []
        self.initLayer(inputSize=inputSize)
    def Add(self):
        newLayer=nlayer(self)
        self.children.append(newLayer)
        newLayer.index=self.children.index(newLayer)
        return newLayer
    def Pop(self):
        return self.children.pop()
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
    def evaluate(self, input, result):
        self.fastfeed(input)
        temp=0
        for i, n in enumerate(self.out().children):
            temp+=(n.sig-result[i])**2
        return temp/len(self.out().children)
    def Activate(self):
        for l in self.children:
            l.ActivateLayer()
    def initLayer(self, inputSize=1):
        self.StackLayer(number=inputSize, function=Normal, defaultSig=1)
    def StackLayer(self, number=1, **kwargs):
        neu=[]
        (defaultSig, function) = (kwargs.get("defaultSig"),kwargs.get("function"))
        function = function if function is not None else Normal
        layer=self.Add()
        for i in range(number):
            n=layer.Add(function)
            n.sig=defaultSig if defaultSig is not None else 1
            n.f=function
            neu.append(n)
        return neu
    def train(self, inputData, result, **kwargs):
        tMethod=kwargs.get("trainMethod")
        tMethod=tMethod if tMethod is not None else Method.GradientDescent
        tMethod(self, inputData, result, arg=kwargs)
    def Copy(self):
        p=nsystem()
        p.ImportModel(self)
        p.__reset__()
        return p
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
    def GetParams(self):
        p=[]
        for l in self.children:
            temp=l.ProvideParam()
            if temp is not None:
                p.append(temp)
        wTable=[]
        bTable=[]
        t=[item for item in [layer["w"] for layer in p]]
        for n in t:
            each=[i["v"] for i in n]
            for e in each:
                for a in e:
                    wTable.append(a)
        t=[item for item in [layer["b"] for layer in p]]
        for n in t:
            each=[i["v"] for i in n]
            for e in each:
                bTable.append(e)
        return (wTable, bTable)
    def SetParams(self, w, b):
        (iw, ib)=(0,0)
        for l in self.children:
            (iw, ib)=l.SetParams(w,b,iw,ib)
    def ImportModel(self, other):
        p=other.Proto()
        self.ImportProto(p)
    def ImportProto(self, other):
        po=other
        if len(self.children)!=po[1]:
            if len(self.children)<po[1]:
                for _ in itertools.repeat(None, po[1]-len(self.children)):
                    self.Add()
            elif len(self.children)>po[1]:
                for _ in itertools.repeat(None, len(self.children)-po[1]):
                    self.Pop()
        for i in po[0]:
            this=self.children[i[1]]
            if len(this.children)!=i[2]:
                if len(this.children)<i[2]:
                    for _ in itertools.repeat(None, i[2]-len(this.children)):
                        this.Add(Normal)
                elif len(this.children)>i[2]:
                    for _ in itertools.repeat(None, len(this.children)-i[2]):
                        this.Pop(Normal)
            for a in i[0]:
                neu=this.children[a[0]]
                neu.f=a[1]
                neu.w=a[2]
                neu.b=a[3]
                neu.sig=a[4]
    def Proto(self):
        t=[]
        for l in self.children:
            t.append(l.Proto())
        return [t, len(self.children)]
    def __reset__(self):
        for l in self.children:
            l.__reset__()
class nlayer:
    def __init__(self, sys=None):
        self.children = []
        self.index = None
        self.sys = sys
    def __reset__(self):
        for n in self.children:
            n.__reset__()
    def Pop(self):
        return self.children.pop()
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
    def Proto(self):
        sender=[]
        for n in self.children:
            (index, f, w, b, sig)=n.Proto()
            sender.append([index, f, w, b, sig])
        return [sender, self.index, len(self.children)]
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
    def ProvideParam(self):
        if self.index <= 0:
            return None
        weights=[]
        biases=[]
        for n in self.children:
            weights.append({"index":n.index, "v":n.w})
            biases.append({"index":n.index, "v":n.b})
        return {"layer":self.index, "w":weights, "b":biases}
    def SetParams(self, w, b, iw, ib):
        if self.index <= 0:
            return (iw, ib)
        indexW=iw
        indexB=ib
        (thisw, thisb)=([], 1)
        for n in self.children:
            for _ in itertools.repeat(None, len(n.w)):
                thisw.append(w[indexW])
                indexW+=1
            thisb=b[indexB]
            indexB+=1
            n.w=thisw
            n.b=thisb
            (thisw, thisb)=([], 1)
        return (indexW, indexB)
class nneuron:
    def __init__(self, layer=None):
        self.layer = layer
        self.index = None
        self.sig = 1
        self.w = [random.uniform(0.5,1) for x in range(len(self.layer.ProvidePrevSig()))]
        self.b = random.uniform(0.5,1)
        self.f = Normal
    def __reset__(self):
        self.sig = 1
        self.w = [random.uniform(0.5,1) for x in range(len(self.layer.ProvidePrevSig()))]
        self.b = random.uniform(0.5,1)
    """index, f, w, b, sig
    """
    def Proto(self):
        return (self.index, self.f, self.w, self.b, self.sig)
    def ActivateEach(self):
        self.sig=0
        self.sig=self.f.subs(x, self.Z()) 
    def ActDiff(self):
        return sympy.diff(self.f)
    def Z(self):
        prev=self.layer.ProvidePrevSig()
        sum=0
        for i, p in enumerate(prev):
            if self.w[i]==None:
                self.w[i]=random.uniform(0.5,1)
            sum+=self.w[i]*p
        return sum+self.b
    def diffSigZ(self):
        return (self.ActDiff().subs(x,self.Z()))
    def ToString(self):
        return f'  ({self.index})sig:{self.sig} weights:{self.w} bias:{self.b}'
    def ToStringOnlySig(self):
        return f'({self.sig})'
class Method:
    @classmethod
    def GradientDescent(cls, sys, inputData, result, arg):
        sys.fastfeed(inputData)
        alpha=arg.get("alpha")
        alpha=alpha if alpha is not None else 0.05
        for i, n in enumerate(sys.out().children):
            cls.GradientDescentInner(n, trainData=result[i], rate=alpha)
    @classmethod
    def GradientDescentInner(cls, this, **kwargs):
        if this.layer.index<=1:
            return
        t=kwargs.get("t")
        rate=kwargs.get("rate")
        train=kwargs.get("trainData")
        if train==None:
            return -1 
        if t==None:
            dt=(2/len(this.layer.children))*(this.sig-train)
        else:
            dt=t
        dt=dt*this.diffSigZ()
        for i, n in enumerate(this.layer.ProvidePrevNeu()):
            provide=dt*this.w[i]
            cls.GradientDescentInner(n, t=provide, trainData=train, rate=rate)
            this.w[i]=this.w[i] - dt*n.sig*rate
        this.b=this.b - dt*rate
    @classmethod
    def Genetic(cls, sys, inputData, result, **kwargs):
        #세대당 개체수v, 돌연변이 확률v, 돌연변이 평균v, 돌연변이 표준편차v, 유전 개체군/랜덤 개체군 퍼센트, 교차 방법, 엘리트 유전자 남기는 비율
        arg=kwargs.get("arg") if kwargs.get("arg") is not None else {}
        size=arg.get("poolSize") if arg.get("poolSize") is not None else 3
        muchance=arg.get("muChance") if arg.get("muChance") is not None else 0.1
        sigma=arg.get("muSigma") if arg.get("muSigma") is not None else 1
        avg=arg.get("muExp") if arg.get("muExp") is not None else 0
        eliteRatio=arg.get("eliteRatio") if arg.get("eliteRatio") is not None else 0.2
        randRatio=arg.get("randRatio") if arg.get("randRatio") is not None else 0.2
        memory=sys.memory
        if not memory:
            for _ in itertools.repeat(None, size):
                memory.append({"model":sys.Copy(), "fitness":0})
            memory.append({"proto":sys.Copy()})
        proto=[i["proto"] for i in memory if "proto" in i][0]
        parentPool=[i for i in memory if "model" in i]
        maxFit=0
        maxModel=proto
        for m in parentPool:
            model=m.get("model")
            if not model:
                continue
            fit=abs(model.evaluate(inputData, result)-1)
            m.update({"fitness":fit})
            if maxFit<fit:
                maxFit=fit
                maxModel=model
        if maxModel is not proto:
            sys.ImportModel(maxModel)
        fTable=[(0 if f.get("fitness") is None else f.get("fitness")) for f in parentPool]
        p=cls.softmax(fTable)
        childPool=[]
        workRatio=1-(eliteRatio+randRatio)
        for _ in itertools.repeat(None, round(size*workRatio)):
            parent=np.random.choice(parentPool, p=p, size=2, replace=True)
            w=[]
            b=[]
            for k in parent:
                (tw, tb)=cls.GeneticGetTable(k)
                if tw is not None and tb is not None:
                    w.append(tw)
                    b.append(tb)
            childW=cls.cross(w[0], w[1], muchance, sigma, avg)
            childB=cls.cross(b[0], b[1], muchance, sigma, avg)
            child=proto.Copy()
            child.SetParams(childW, childB)
            childPool.append({"model":child, "fitness":0})
        for _ in itertools.repeat(None, round(size*randRatio)):
            child=proto.Copy()
            childPool.append({"model":child, "fitness":0})
        parent=np.random.choice(parentPool, p=p, size=round(size*eliteRatio), replace=True)
        for p in parent:
            childPool.append({"model":p.get("model"), "fitness":p.get("fitness")})
        childPool.append({"proto":proto})
        sys.memory=childPool
    @staticmethod
    def GeneticGetTable(e):
        model=e.get("model")
        if not model:
            return (None, None)
        return model.GetParams()

    @staticmethod
    def softmax(x):
        s=sum([math.exp(i) for i in x])
        return [math.exp(i)/s for i in x]
    @staticmethod
    def cross(x :list, y :list, muchance :float, sigma :float, avg :float) -> list:
        if len(x)!=len(y):
            print(f"Length does not match -> {len(x)} {len(y)}")
            raise(f"Length does not match -> {len(x)} {len(y)}")
        l=len(x)
        dvid=random.randint(0, l)
        result=[*x[:dvid], *y[dvid:]]
        for i, a in enumerate(result):
            if random.uniform(0, 1) < muchance:
                result[i]+=random.gauss(avg,sigma)
        return result
    
        
Normal=x
ReLU=sympy.Max(x, 0)
Sigmoid=1/(1+sympy.exp(x))



