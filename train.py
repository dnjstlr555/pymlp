import mlp
import random
import csv
import pickle

(male, female) = (0,1)
(ga, gb, gc, gd, ge) = (0,1,2,3,4)
(associate, highsch, somecollge, somehighsch, bachelor) = (0,1,2,3,4)
(standard, reduced) = (0, 1)
class people:
    gender=male
    race=ga
    pedu=highsch
    lunch=standard
    test=0
    math=75
    read=75
    write=75
    def ToString(self):
        return f"T:{self.test} gender={self.gender}, race={self.race}, parentalEducation={self.pedu}, lunch={self.lunch}, math={self.math}, read={self.read}, write={self.write}"
    def ToListOnlyInfo(self):
        return [self.gender, self.race+random.gauss(0,0.5), self.pedu+random.gauss(0,0.5), self.lunch+random.gauss(0,0.5), self.math+random.gauss(0,0.5), self.read+random.gauss(0,0.5), self.write+random.gauss(0,0.5)]
    def IsTestOkay(self):
        return self.test
dataset=[]
testset=[]
def converter(row):
    p=people()
    if row[0]!="male" and row[0]!="female":
        return None
    p.gender=male if row[0]=="male" else female
    if row[1]=="group A":
        p.race=ga
    elif row[1]=="group B":
        p.race=gb
    elif row[1]=="group C":
        p.race=gc
    elif row[1]=="group D":
        p.race=gd
    elif row[1]=="group E":
        p.race=ge
    if row[2]=="some college":
        p.pedu=somecollge
    elif row[2]=="associate's degree":
        p.pedu=associate
    elif row[2]=="high school":
        p.pedu=highsch
    elif row[2]=="some high school":
        p.pedu=somehighsch
    elif row[2]=="bachelor's degree":
        p.pedu=bachelor
    p.lunch=standard if row[3]=="standard" else reduced
    p.test=0 if row[4]=="none" else 1
    p.math=int(row[5])
    p.read=int(row[6])
    p.write=int(row[7])
    return p

with open('train.csv', 'r', encoding='utf-8', newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        c=converter(row)
        if c is not None:
            dataset.append(c)
with open('test.csv', 'r', encoding='utf-8', newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        c=converter(row)
        if c is not None:
            testset.append(c)

model=mlp.nsystem(inputSize=7)
model.StackLayer(number=8, function=mlp.ReLU)
model.StackLayer(number=8, function=mlp.ReLU)
model.StackLayer(number=1, function=mlp.Sigmoid)

random.shuffle(dataset)
for i, d in enumerate(dataset):
    model.train(d.ToListOnlyInfo(), [d.IsTestOkay()], trainMethod=mlp.Method.GradientDescent, alpha=0.001)
    if i%50==0:
        count=1
        sum=0
        for t in testset:
            sum+=model.evaluate(t.ToListOnlyInfo(), [t.IsTestOkay()])
            print(model.fastfeed(t.ToListOnlyInfo()))
            count+=1
        print(sum/count)
f = open('store2.pckl', 'wb')
pro=model.Proto()
pickle.dump(pro, f)
f.close()
