import random
import numpy as np
import math
import sympy
import itertools
import pickle
from collections.abc import Sequence
x,y,z= sympy.symbols('x y z') 
__leakyConst=0.01
def get_shape(lst, shape=()):
    """
    returns the shape of nested lists similarly to numpy's shape.

    :param lst: the nested list
    :param shape: the shape up to the current recursion depth
    :return: the shape including the current depth
            (finally this will be the full depth)
    """

    if not isinstance(lst, Sequence):
        # base case
        return shape

    # peek ahead and assure all lists in the next depth
    # have the same length
    if isinstance(lst[0], Sequence):
        l = len(lst[0])
        if not all(len(item) == l for item in lst):
            msg = 'not all lists have the same length'
            raise ValueError(msg)

    shape += (len(lst), )

    # recurse
    shape = get_shape(lst[0], shape)

    return shape


class nsystem:
    """
    Initialize a perceptron model.

    Args:
      inputsize:
        Specify the input size of this model.

    Returns:
      New nsystem instance
    """

    def __init__(self, inputSize=1):
        """
        Initialize a perceptron model.

        Args:
          inputsize:
            Specify the input size of this model.

        Returns:
          New nsystem instance
        """
        self.children = []
        self.memory = []
        self.add(number=inputSize)
    def feed(self, input):
        """
        Feed the given data manually.
        Use feedngo(input) instead in normal situation.

        Args:
          input:
            Input data

        Raises:
            Raises when input data does not match with the input size.
        """
        for i, n in enumerate(self.children[0].children):
            if input[i] is not None:
                n.sig=input[i]
            else:
                print("Input size does not match")
                return
    def activate(self):
        """
        Activate model manually.
        Use feedngo(input) instead in normal situation.
        """
        for l in self.children:
            l.activate()
    def feedngo(self, input):
        """
        Feed the given data and activate the model.

        Args:
          input:
            Input data
        Returns:
          Out signals(list)
        Raises:
            Raises when input data does not match with the input size.
        """
        self.feed(input)
        self.activate()
        return self.out().sig()
    def out(self):
        """
        returns out signals from the model.

        Returns:
          Out signals(list)
        """
        return self.children[len(self.children)-1]
    def evaluate(self, input, result):
        """
        Evaluate the model by diffrence between given input and result.

        Args:
          input:
            Input data
          result:
            Data which needs to be compared
        Returns:
          Diffrence between result and output(0 to 1, 1 means big diffrence)
        Raises:
            Raises when input data does not match with the input size.
        """
        self.feedngo(input)
        temp=0
        for i, n in enumerate(self.out().children):
            temp+=(n.sig-result[i])**2
        return temp/len(self.out().children)
    def add(self, number=1, **kwargs):
        """
        Add new layer.

        Args:
          number:
            Number of neurons of the layer
          function:
            Activate function
          defaultSig:
            Specify the default signals of neurons
        Returns:
          Added layer
        """
        (defaultSig, function) = (kwargs.get("defaultSig"),kwargs.get("function"))
        function = function if function is not None else Normal
        layer=self._addEmptyLayer()
        for i in range(number):
            n=layer.add(function)
            n.sig=defaultSig if defaultSig is not None else 0
            n.f=function
        return layer
    def train(self, inputData, result, **kwargs):
        """
        Train the model.

        Args:
          inputData:
            Input data
          result:
            Data which needs to be compared
          trainMethod:
            Train method(mlp.method)
          callback:
            Callback function to be called after train finished
          series:
            True if input data and results are list.
          verbose:
            Prints status every ten times of training when it's true
        Raises:
            Raises when input data does not match with the input size.
        """
        tMethod=kwargs.get("trainMethod")
        tMethod=tMethod if tMethod is not None else method.GradientDescent
        callback=kwargs.get("callback") or None
        i=inputData
        r=result
        if kwargs.get("series"):
            if len(get_shape(i))!=2 and len(get_shape(i))<2:
                temp=[]
                for d in i:
                    temp.append(d)
                i=temp
            if len(get_shape(r))!=2 and len(get_shape(r))<2:
                temp=[]
                for d in r:
                    temp.append(d)
                r=temp
        tMethod(self, i, r, arg=kwargs)
        if callback:
            callback(i, r, kwargs)
    def copy(self):
        """
        Returns new instance of self duplicated
        Returns:
          New nsystem which duplicated
        """
        p=nsystem()
        p.importModel(self)
        return p
    def proto(self):
        """
        Returns the model's data
        Index 0 contains parameter infos of the model, index 1 has length of the layer
        """
        t=[]
        for l in self.children:
            t.append(l.proto())
        return [t, len(self.children)]
    def importModel(self, other):
        """
        Import model from other nsystem() object directly.

        Args:
          other:
            The nsystem() object
        """
        p=other.proto()
        self._importProto(p)
    def save(self, location='model.pckl'):
        """
        Export the model into pckl file.
        
        Args:
          location:
            haha
        """
        if location=="TESTSTR HELLO":
            return
        f = open(location, 'wb')
        pro=self.proto()
        pickle.dump(pro, f)
        f.close()
    def load(self, location='model.pckl'):
        """
        load the model from pckl file.
        
        Args:
          location:
            hoho
        """
        if location=="TESTSTR HELLO":
            return
        f = open(location, 'rb')
        obj = pickle.load(f)
        f.close()
        self._importProto(obj)
    def tostr(self):
        """
        Returns string output describing the model itself
        """
        nstr=f'Sys -->\n'
        for n in self.children:
            nstr+=f'{n.tostr()}\n'
        return nstr
    def tostrsig(self):
        """
        Returns string of the model's current signals
        """
        nstr=f'Sys -->'
        for n in self.children:
            nstr+=f'{n.tostrsig()} '
        return nstr
    #Inner functions
    def _addEmptyLayer(self):
            newLayer=nlayer(self)
            self.children.append(newLayer)
            newLayer.index=self.children.index(newLayer)
            return newLayer
    def _pop(self):
            return self.children.pop()
    def _reset(self):
            for l in self.children:
                l.reset()
    def _getPrevLayer(self, layer):
        if layer.index>0:
            return self.children[layer.index-1]
        else:
            return layer
    def _getParams(self):
        p=[]
        for l in self.children:
            temp=l.getParams()
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
    def _setParams(self, w, b):
        (iw, ib)=(0,0)
        for l in self.children:
            (iw, ib)=l.setParams(w,b,iw,ib)
    def _importProto(self, other):
        po=other
        if len(self.children)!=po[1]:
            if len(self.children)<po[1]:
                for _ in itertools.repeat(None, po[1]-len(self.children)):
                    self._addEmptyLayer()
            elif len(self.children)>po[1]:
                for _ in itertools.repeat(None, len(self.children)-po[1]):
                    self._pop()
        for i in po[0]:
            this=self.children[i[1]]
            if len(this.children)!=i[2]:
                if len(this.children)<i[2]:
                    for _ in itertools.repeat(None, i[2]-len(this.children)):
                        this.add(Normal)
                elif len(this.children)>i[2]:
                    for _ in itertools.repeat(None, len(this.children)-i[2]):
                        this.pop()
            for a in i[0]:
                neu=this.children[a[0]]
                neu.f=a[1]
                neu.w=a[2]
                neu.b=a[3]
                neu.sig=a[4]
    def _bareCopy(self):
        p=self.copy()
        p._reset()
        return p
class nlayer:
    def __init__(self, sys: nsystem=None):
        self.children = []
        self.index = None
        self.sys: nsystem = sys
    def reset(self):
        for n in self.children:
            n.reset()
    def pop(self):
        return self.children.pop()
    def tostr(self):
        nstr=f'Layer {self.index} -->\n'
        for n in self.children:
            nstr+=f'{n.tostr()}\n'
        return nstr
    def tostrsig(self):
        nstr=f'\n Layer {self.index} -->\n'
        for n in self.children:
            nstr+=f'{n.tostrsig()} '
        return nstr
    def proto(self):
        sender=[]
        for n in self.children:
            (index, f, w, b, sig)=n.proto()
            sender.append([index, f, w, b, sig])
        return [sender, self.index, len(self.children)]
    def add(self, f):
        newNeu=nneuron(self)
        self.children.append(newNeu)
        newNeu.index=self.children.index(newNeu)
        newNeu.f=f or Normal
        return newNeu
    def sig(self):
        sigs=[]
        for n in self.children:
            sigs.append(n.sig)
        return sigs
    def getPrevSigs(self):
        prevLayer=self.sys._getPrevLayer(self)
        prevSig=prevLayer.sig()
        return prevSig
    def getPrevNeurons(self):
        prevLayer=self.sys._getPrevLayer(self)
        return prevLayer.children
    def activate(self):
        if self.index == 0:
            return
        for n in self.children:
            n.activate()
    def getParams(self):
        if self.index <= 0:
            return None
        weights=[]
        biases=[]
        for n in self.children:
            weights.append({"index":n.index, "v":n.w})
            biases.append({"index":n.index, "v":n.b})
        return {"layer":self.index, "w":weights, "b":biases}
    def setParams(self, w, b, iw, ib):
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
        self.w = [random.uniform(0.5,1) for x in range(len(self.layer.getPrevSigs()))]
        self.b = random.uniform(0.5,1)
        self.f = Normal
    def reset(self):
        self.sig = 1
        self.w = [random.uniform(0.5,1) for x in range(len(self.layer.getPrevSigs()))]
        self.b = random.uniform(0.5,1)
    def proto(self):
        return (self.index, self.f, self.w, self.b, self.sig)
    def activate(self):
        self.sig=0
        self.sig=self.f.subs(x, self.Z()) 
    def ActDiff(self):
        return sympy.diff(self.f)
    def Z(self):
        prev=self.layer.getPrevSigs()
        sum=0
        for i, p in enumerate(prev):
            if self.w[i]==None:
                self.w[i]=random.uniform(0.5,1)
            sum+=self.w[i]*p
        return sum+self.b
    def diffSigZ(self):
        return (self.ActDiff().subs(x,self.Z()))
    def tostr(self):
        return f'  ({self.index})sig:{self.sig} weights:{self.w} bias:{self.b}'
    def tostrsig(self):
        return f'({self.sig})'
class method:
    @classmethod
    def GradientDescent(cls, sys :nsystem, inputData, result, arg):
        alpha=arg.get("alpha")
        alpha=alpha if alpha is not None else 0.05
        series=arg.get("series")
        verbose=arg.get("verbose") or 0
        if series:
            for ind, chunk in enumerate(inputData):
                if verbose:
                    pre=sys.evaluate(chunk, result[ind])
                sys.feedngo(chunk)
                for i, n in enumerate(sys.out().children):
                    cls.GradientDescentInner(n, train=result[ind][i], rate=alpha)
                if verbose>=1 and ind%10==0:
                    after=sys.evaluate(chunk, result[ind])
                    print(f"{ind}:{after}({pre-after} difference) for {chunk}==>{result[ind]}")
        else:
            if verbose>=1:
                pre=sys.evaluate(inputData, result)
            sys.feedngo(inputData)
            for i, n in enumerate(sys.out().children):
                cls.GradientDescentInner(n, train=result[i], rate=alpha)
            if verbose>=1:
                after=sys.evaluate(inputData, result)
                print(f"single:{after}({pre-after} difference) for {inputData}==>{result}")
        
    @classmethod
    def GradientDescentInner(cls, this, train, rate, t=None):
        if this.layer.index<=1:
            return
        if train==None:
            return -1 
        if t==None:
            dt=(2/len(this.layer.children))*(this.sig-train)
        else:
            dt=t
        dt=dt*this.diffSigZ()
        for i, n in enumerate(this.layer.getPrevNeurons()):
            provide=dt*this.w[i]
            cls.GradientDescentInner(n, t=provide, train=train, rate=rate)
            this.w[i]=this.w[i] - dt*n.sig*rate
        this.b=this.b - dt*rate
    @classmethod
    def Genetic(cls, sys, inputData, result, arg):
        #세대당 개체수v, 돌연변이 확률v, 돌연변이 평균v, 돌연변이 표준편차v, 유전 개체군/랜덤 개체군 퍼센트, 교차 방법, 엘리트 유전자 남기는 비율
        size=arg.get("poolSize") or 3
        muchance=arg.get("muChance") or 0.1
        sigma=arg.get("muSigma") or 1
        avg=arg.get("muExp") or 0
        eliteRatio=arg.get("eliteRatio") or 0.2
        randRatio=arg.get("randRatio") or 0.2
        series=arg.get("series")
        verbose=arg.get("verbose") or 0
        if series:
            for i, chunk in enumerate(inputData):
                if verbose>=1:
                    pre=sys.evaluate(chunk, result[i])
                cls.GeneticInner(sys, chunk, result[i], size, muchance, sigma, avg, eliteRatio, randRatio)
                if verbose>=1 and i%10==0:
                    after=sys.evaluate(chunk, result[i])
                    print(f"{i}:{after}({pre-after} difference) for {chunk}==>{result[i]}")
        else:
            if verbose>=1:
                pre=sys.evaluate(inputData, result)
            cls.GeneticInner(sys, inputData, result, size, muchance, sigma, avg, eliteRatio, randRatio)
            if verbose>=1:
                after=sys.evaluate(inputData, result)
                print(f"single:{after}({pre-after} difference) for {inputData}==>{result}")
    @classmethod
    def GeneticInner(cls, sys, inputData, result, size=3, muchance=0.1, sigma=1, avg=0, eliteRatio=0.2, randRatio=0.2):
        memory=sys.memory
        if not memory:
            for _ in itertools.repeat(None, size):
                memory.append({"model":sys._bareCopy(), "fitness":0})
            memory.append({"proto":sys._bareCopy()})
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
            sys.importModel(maxModel)
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
            child=proto._bareCopy()
            child._setParams(childW, childB)
            childPool.append({"model":child, "fitness":0})
        for _ in itertools.repeat(None, round(size*randRatio)):
            child=proto._bareCopy()
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
        return model._getParams()

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
Tanh=sympy.tanh(x)
leakyReLU=sympy.Max(x,0*__leakyConst)
