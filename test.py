import mlp
import traceback
import random
#Function tests
sys=mlp.nsystem(inputSize=2)
sys.add(number=3, function=mlp.ReLU)
layer=sys.add(number=3, function=mlp.ReLU)
sys.add(number=3, function=mlp.ReLU)
try:
    sys.feed([1,1])
    sys.activate()
    r=sys.feedngo([1,1])
    r=sys.out()
    r=sys.evaluate([1,1], [1,1,1])
    sys.train([1,1], [1,1,1])
    others=sys.copy()
    sample=sys.proto()
    sys.importModel(others)
    sys.save("TESTSTR HELLO")
    sys.load("TESTSTR HELLO")
    sys.tostr()
    sys.tostrsig()
    sys._addEmptyLayer()
    sys._pop()
    sys._reset()
    (ww, bb)=sys._getParams()
    sys._getPrevLayer(layer)
    sys._setParams(ww,bb)
    sys._importProto(sample)
except Exception as e:
    print(f"-----")
    traceback.print_exc()
    print(f"-- nsystem Function error occured --")
    
else:
    print("nsystem Function passed")

#Common tests

new=sys.copy()
new.importModel(sys)
print(f"ImportModel Works:{sys.tostr()==new.tostr()}")

sys=mlp.nsystem(inputSize=2)
sys.add(number=3, function=mlp.ReLU)
sys.memory.append({"model":sys.copy(), "fitness":0})
sys.memory.append({"model":sys.copy(), "fitness":0})
parentPool=[i for i in sys.memory if "model" in i]
origin=parentPool[0].get("model")

(tw, tb)=mlp.method.GeneticGetTable(parentPool[0])
(tw2, tb2)=mlp.method.GeneticGetTable(parentPool[1])
newW=mlp.method.cross(tw, tw2, 0.01, 1, 0)
newB=mlp.method.cross(tb, tb2, 0.01, 1, 0)
str=f"origin: {len(tw)} {len(tb)} ->after: {len(newW)} {len(newB)} (error)"
print(f"Genetic Cross length validate: {True if (len(tw)==len(newW) and len(tb)==len(newB)) else str}")

#setparm
new=sys.copy()
new._reset()
(w1, b1)=sys._getParams()
(w2, b2)=new._getParams()
print(f"Children Deep Copied check: {True if w1!=w2 else False}")

sys._setParams(w1,b1)
(aw, ab)=sys._getParams()
print(f"Set Param check for itself: {True if aw==w1 else False}")
new._setParams(w1,b1)
(aw, ab)=new._getParams()
print(f"Set Param check for copy: {True if aw==w1 else False}")

sys=mlp.nsystem(inputSize=2)
sys.add(number=3, function=mlp.ReLU)
sys.add(number=4,function=mlp.Sigmoid)
(before, _)=sys._getParams()
sys.train([random.uniform(-1,1),random.uniform(-1,1)], [random.uniform(-1,1) for _ in range(4)], trainMethod=mlp.method.GradientDescent)
print(f"Gradient Trained check: {before!=sys._getParams()[0]}")

srin=[]
srout=[]
for i in range(3):
    srin.append([random.uniform(-1,1),random.uniform(-1,1)])
    srout.append([random.uniform(-1,1) for _ in range(4)])
(before, _)=sys._getParams()
sys.train(srin, srout, series=True, trainMethod=mlp.method.GradientDescent)
print(f"Gradient Series Trained check: {before!=sys._getParams()[0]}")

(before, _)=sys._getParams()
sys.train([random.uniform(-1,1),random.uniform(-1,1)], [random.uniform(-1,1) for _ in range(4)], trainMethod=mlp.method.Genetic)
print(f"Genetic Trained check: {before!=sys._getParams()[0]}")

(before, _)=sys._getParams()
sys.train(srin, srout, series=True, trainMethod=mlp.method.Genetic)
print(f"Genetic Series Trained check: {before!=sys._getParams()[0]}")