import mlp


md=model
md.add(mlp.layer.fc(3),
    mlp.layer.fc(2),
    mlp.layer.fc(4)) #stack
md.add(mlp.layer.fc(3),
    mlp.layer.fc(2),
    mlp.layer.fc(4), parallel=True)
md.add((mlp.layer.fc(3),
    mlp.layer.fc(2),
    mlp.layer.fc(4))) #tuple
i=mlp.layer.input(2)
l1=mlp.layer.fc(3)
l1_1=mlp.layer.fc(3)
l2=mlp.layer.fc(4)
l3=mlp.layer.fc(2)
l1.conin(i)
l1_1.conin(i)
l2.conin(l1)
l2.conin(l1_1)
l3.conin(l2)
i.set([3,2])
i.act()
for d in i.out():
    print(d.sig)
for d in l1.out():
    print(d.sig)
for d in l1_1.out():
    print(d.sig)
for d in l2.out():
    print(d.sig)
for d in l3.out():
    print(d.sig)
