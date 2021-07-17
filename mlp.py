import numpy as np

# todo : nbundle을 child에 바로 귀속시키는게 아니라 일종의 planner 로써로만 사용되게. 
class nmodel:
    def __init__(self):
        pass
    def add(self, *args, **kwargs):
        parallel=kwargs.get("parallel")

        for i in args:
            if parallel:
                
            else:
        pass


class nlayer:
    def __init__(self, n):
        self.type="Placeholder"
        self.desc="default layer"
        self.child=[]
        self.ninput=[]
        self.nout=[]
    def add(self, bundleType, n):
        new=bundleType(self)
        for _ in range(n):
            new.add()
        self.child.append(new)
    def out(self):
        """Returns child cells array.
        
        :returns: An array of cells belong to self childs
        """
        o=[]
        for i in self.child:
            for j in i.out():
                o.append(j)
        return o
    def get(self):
        """Returns cells of input layers attached to this layer
        
        :returns: An array of cells belong to input layers
        """
        i=[]
        for j in self.ninput:
            for k in j.out():
                i.append(k)
        return i
    def conin(self, other):
        """Connect other layer into this layer as input
        
        :param nlayer other: nlayer 
        """
        self.ninput.append(other)
        other.__sigin(self)
        for i in self.child:
            for j in i.cells:
                j.init()
    def discon(self, other):
        """Disconnect other layer that previously connected 
        
        :param nlayer other: nlayer 
        """
        self.ninput.remove(other)
        other.__sigdiscon(self)
    def __sigin(self, other):
        """Get signals other's registration to this layer as input and register other as out
        
        :param nlayer other: nlayer 
        """
        self.nout.append(other)
    def __sigdiscon(self, other):
        """Remove other from out array
        
        :param nlayer other: nlayer 
        """
        self.nout.remove(other)
    def act(self):
        """Activate self and before out layers
        """
        self.dev("Activation Start")
        i=self.get()
        for j in self.out():
            j.act(i)
        for oth in self.nout:
            oth.act()
    def grad(self):
        pass
    def train(self):
        pass
    def dev(self, msg, *args):
        print(f"{id(self)}:{self.type} - {msg}")
            
class nbundle:
    def __init__(self, p):
        self.type="Placeholder bundle"
        self.desc="default bundle"
        self.parent=p
        self.cells=[]
    def out(self):
        """Returns cells of the bundle
        
        :returns: An array of cells belong to this bundle
        """
        return self.cells
    def add(self):
        new=cell(self.parent, self.act, self.grad, self.init)
        self.cells.append(new)
    @staticmethod
    def act(p, x):
        pass
    @staticmethod
    def grad(p, x):
        pass
    @staticmethod
    def init(p):
        pass

class cell:
    def __init__(self, p, act, grad, init):
        self.actBehave=act or self.__actProto
        self.gradBehave=grad or self.__gradProto
        self.initBehave=init or self.__initProto
        self.parent=p
        self.sig=0
    def init(self):
        self.initBehave(self)
    def inject(self, act, grad, init):
        self.actBehave=act
        self.gradBehave=grad
    def act(self, x):
        self.actBehave(self, x)
    def grad(self, x):
        self.gradBehave(self, x)
    @staticmethod
    def __actProto(p, x):
        pass
    @staticmethod
    def __gradProto(p, x):
        pass
    @staticmethod
    def __initProto(p):
        pass

class bundle:
    class input(nbundle):
        def __init__(self, p):
            super().__init__(p)
            self.type="Input Bundle"
            self.desc="Input"
    class fc(nbundle):
        def __init__(self, p):
            super().__init__(p)
            self.type="FC Bundle"
            self.desc="Basic fully connected bundle"
        @staticmethod
        def init(p):
            p.w=[1]*len(p.parent.get())
            p.b=0
        @staticmethod
        def act(p, x):
            sigs=[i.sig for i in x]
            sum=np.dot(sigs,np.transpose(p.w))
            p.sig=sum+p.b

class layer:
    class input(nlayer):
        def __init__(self, n):
            super().__init__(n)
            self.type="Input Layer"
            self.desc="Input"
            self.add(bundle.input, n)
        def set(self, x):
            th=self.out()
            for i, c in enumerate(th):
                c.sig=x[i]
    class fc(nlayer):
        def __init__(self, n):
            super().__init__(n)
            self.type="FC Layer"
            self.desc="Basic fully connected layer"
            self.add(bundle.fc, n)