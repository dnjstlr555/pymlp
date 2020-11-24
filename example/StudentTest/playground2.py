import mlp
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
        return [self.gender, self.race, self.pedu, self.lunch, self.math, self.read, self.write]
    def IsTestOkay(self):
        return self.test


model=mlp.nsystem()
model.load(location='store2.pckl')
print(model.ToString())

me=people()
me.gender=male
me.race=ga
me.pedu=bachelor
me.lunch=standard
me.math=36
me.read=72
me.write=100
print(model.fastfeed(me.ToListOnlyInfo()))