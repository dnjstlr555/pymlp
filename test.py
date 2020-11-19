import mlp
import random

sys=mlp.nsystem(inputSize=2)
sys.StackLayer(number=3, function=mlp.ReLU)
sys.StackLayer(number=3, function=mlp.ReLU)
sys.StackLayer(number=3, function=mlp.ReLU)
new=sys.Copy()
new.ImportModel(sys)
print(f"ImportModel Works:{sys.ToString()==new.ToString()}")

sys=mlp.nsystem(inputSize=2)
sys.StackLayer(number=3, function=mlp.ReLU)
sys.memory.append({"model":sys.Copy(), "fitness":0})
sys.memory.append({"model":sys.Copy(), "fitness":0})
parentPool=[i for i in sys.memory if "model" in i]
origin=parentPool[0].get("model")

(tw, tb)=mlp.Method.GeneticGetTable(parentPool[0])
(tw2, tb2)=mlp.Method.GeneticGetTable(parentPool[1])
newW=mlp.Method.cross(tw, tw2, 0.01, 1, 0)
newB=mlp.Method.cross(tb, tb2, 0.01, 1, 0)
str=f"origin: {len(tw)} {len(tb)} ->after: {len(newW)} {len(newB)} (error)"
print(f"Genetic Cross length validate: {True if (len(tw)==len(newW) and len(tb)==len(newB)) else str}")

#setparm
new=sys.Copy()
(w1, b1)=sys.GetParams()
(w2, b2)=new.GetParams()
print(f"Children Deep Copied check: {True if w1!=w2 else False}")

sys.SetParams(w1, b1)
(aw, ab)=sys.GetParams()
print(f"Set Param check for itself: {True if aw==w1 else False}")
new.SetParams(w1, b1)
(aw, ab)=new.GetParams()
print(f"Set Param check for copy: {True if aw==w1 else False}")