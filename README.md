# pymlp
```python
import mlp
model=mlp.nsystem(inputSize=2)
model.StackLayer(number=4, function=mlp.ReLU) 
model.StackLayer(number=1, function=mlp.Sigmoid)
x=[[0.03, 0.01], [0.9, 0.97]]
y=[[0], [1]]
for i, data in enumerate(x):
  model.train(inputData=data, result=y[i], trainMethod=mlp.Method.GradientDescent, alpha=0.01)

print(model.fastfeed([0.01, 0.01]))
```
# 1
학습 기능 외 구현
## nsystem
레이어를 추가하고 입출력 기능
```python
sys=nsystem()
layer1=sys.Add() #레이어 추가
sys.feed([1,0]) #입력 설정
sys.Activate() #활성화
result=sys.out() #마지막 레이어의 결과값
print(result)
```
## nlayer
뉴런 관리, 이전 레이어의 뉴런 불러오기
```python
from sympy import *
x=Symbol('x')

ReLU=Max(0,x)
sys=nsystem()
layer1=sys.Add()
layer1.Add(ReLU) #입력된 함수를 활성 함수로 하는 뉴런 추가
layer1.GetAllSig() #해당 레이어의 모든 신호 받기
layer1.ProvidePrevSig() #이전 레이어의 모든 신호
```

## nneuron
```python
from sympy import *
x=Symbol('x')

ReLU=Max(0,x)
sys=nsystem()
layer1=sys.Add()
layer1.Add(ReLU)
for n in layer1.children:
  print(n.w) #이전 레이어의 i번째의 가중치에 해당하는 w[i]를 가진 배열
  print(n.ProvideZ()+n.b) #w[i]*이전 레이어의 i번째 신호를 더한 ProvideZ()
  n.ActivateEach() #위의 연산을 처리하고 n.sig에 저장
```

# 2
학습하기

## 오차역전법
은닉층이 한개면 구하는게 쉬웠는데 여러개일때와 이를 일반화 하는게 어려웠음<br>
처음엔 맨끝부터 시작해 각 뉴런을 확인하면서 변수에 저장하는 방법을 썼는데 식이 너무 어려워지고 메모리도 너무 많이 사용<br>
<img src="https://user-images.githubusercontent.com/21963949/96717512-8b442380-13e1-11eb-87d8-7f9de06baa86.jpg" height="1000">
## 재귀함수
그냥 너무 머리아프니까 현재 뉴런에서 이전 뉴런들에게 정보를 제공하면서 접속하는 재귀함수 형식을 생각해봄<br>
특히 은닉층으로 점점 들어갈수록 필요한 정보들을 앞에서 미리 계산해 놓을수 있었기 때문에 이 방식을 사용<br>
식이 맞는지는 아직도 모르겠음...<Br>
<img src="https://user-images.githubusercontent.com/21963949/96717510-897a6000-13e1-11eb-99fb-1033d2a1c5b9.jpg" height="1000">
```python
class nneuron:
    def ProvideZ(self): #sigma(이전 레이어 값들*W)+B
        prevSig=self.layer.ProvidePrevSig()
        providz=0
        for i, sig in enumerate(prevSig):
            if self.w[i] is None:
                self.w[i]=1 #random.uniform(,1)
            providz+=self.w[i]*sig
        return providz+self.b
    def deltaHZ(self): #Z값에 따른 Output의 변화율
        return diff(self.f).subs(x,self.ProvideZ())
    def back(self, **kwargs):
        if self.layer.index<=0: return 0 #입력층일때는 취소
        temp=0
        t=None
        predict=[]
        for key, value in kwargs.items():
            if key=="predict": #대조 값
                predict=value
            elif key=="t": #재귀 함수로 전달해줄 값
                t=value
        if predict is None: return -1 
        if t is None: #처음에는 t를 제공하지 않음. 출력층으로 간주하고 초기 식 설정
            temp=2/len(self.layer.children)*(self.sig-predict)*(self.deltaHZ()) #(d비용함수/d결과값)*(d결과값/dZ값(ProvideZ))
        else:
            temp=t*self.deltaHZ()  #d비용함수/dW,dB를 구해야 하므로 이전에서 넘어온 temp * dZ값/....? 쓰다보니까 오류 발견 -> 아래 문단으로 ㄲㄱ
        for i, n in enumerate(self.layer.ProvidePrevNeu()):
            self.w[i]-=temp*n.sig*0.1 #아무튼 이전 뉴런에 대해 각각 temp(d비용함수/dZ값)*(dZ값/dW[i])= temp*해당 뉴런의 신호값. 이걸 빼줌 0.1은 학습률 
            n.back(predict=predict, t=temp) #맨 처음 출력층일때 temp에는 d비용함수/dZ 이므로, dZ/d(이전 뉴런의 출력값에 대한 변화율) 을 곱해야함.
        self.b-=temp*0.1 #b는 이전 뉴런들과 무관
```

## 문제점 여러개
### 학습률
처음에는 식을 그냥 그대로 사용했습니다.
```python
self.w[i]-=temp*n.sig 
```
그런데 하다 보니 너무 변화가 민감하고 결과적으로는 아주 살짝만 달라도 전혀 예측하지 않는 결과가 나와서 0.1의 상수를 곱하니까 잘 되었어요
```python
self.w[i]-=temp*n.sig*0.1
```
헌재도 너무 많이 하면 뭔가 너무 심각해지기 때문에 유동적인 학습률 설정이 필요할듯
```python
for i in range(500):
    train(sys, [random.uniform(0,0.3),random.uniform(0,0.3)], [0])
for i in range(500):
    train(sys, [random.uniform(0.7,1),random.uniform(0.7,1)], [1])

print(sys.fastfeed([0.5,0.5]))
print(sys.fastfeed([0.1,0.1]))
print(sys.fastfeed([1,1]))
```
```
dnjstlr555@owonsig-ui-MacBook-Pro ~ % /usr/local/bin/python /Users/dnjstlr555/Documents/ml/study/fastper.py
[0.00896908128147911] (0.5, 0.5)
[0.256070057435812] (0.1, 0.1)
[0] (1, 1)

학습률 적용후
dnjstlr555@owonsig-ui-MacBook-Pro ~ % /usr/local/bin/python /Users/dnjstlr555/Documents/ml/study/fastper.py
[0.751231125022578] (0.5, 0.5)
[0.379207111783163] (0.1, 0.1)
[1.12325513826199] (1, 1)
```
### 학습 순서
의외로 어떤 데이터를 먼저 학습하고 하지 말아야 했는지가 중요했어요
처음에는 그냥 따로따로 섞지 않고 했었는데 그러니까 제대로 결과가 안나왔는데
둘이 섞어서 랜덤으로 학습하니까 개선된 모습이 나왔습니다. 근데 아직도 들쑥 날쑥하네요..
```python
for i in range(500):
    rand = random.choice((0,1))
    if rand==0:
        train(sys, [random.uniform(0,0.3),random.uniform(0,0.3)], [0])
    else:
        train(sys, [random.uniform(0.7,1),random.uniform(0.7,1)], [1])
```

```
dnjstlr555@owonsig-ui-MacBook-Pro ~ % /usr/local/bin/python /Users/dnjstlr555/Documents/ml/study/fastper.py
[0] 0.5 0.5
[0] 0.1 0.1
[0] 1 1
dnjstlr555@owonsig-ui-MacBook-Pro ~ % /usr/local/bin/python /Users/dnjstlr555/Documents/ml/study/fastper.py
[0.443416099077919] 0.5 0.5
[0] 0.1 0.1
[1.27199433145730]  1 1
```
### 식 오류
글을 쓰다 보니까 식에 문제가 있었어요. 원래 temp 자체 값은 d비용함수/dZ값 변화율로 끝나는데 여기다 원래는 dz값/d넘겨줄 대상의 결과값 변화율을 곱해야 합니다<br>
```python
for i, n in enumerate(self.layer.ProvidePrevNeu()):
    provid=temp*self.w[i]
    self.w[i]-=temp*n.sig*0.1
    n.back(predict=predict, t=provid)
```
그러나 이때 미분하면 대응하는 w값을 곱해줘야 하는게 나오는데 변화된 새 w값을 곱할지 기존 w값을 곱할지 잘 모르겠음. 일단 둘의 큰 차이는 안느껴짐
```
dnjstlr555@owonsig-ui-MacBook-Pro ~ % /usr/local/bin/python /Users/dnjstlr555/Documents/ml/study/fastper.py
Sys -->
Layer 0 -->
  (0)sig:0.04925212691183812 weights:[] bias:0
  (1)sig:0.011959374114055276 weights:[None] bias:0

Layer 1 -->
  (0)sig:0 weights:[0.712259965793124, 0.744655776053618] bias:-0.306526353649786
  (1)sig:0 weights:[0.712259965793124, 0.744655776053618] bias:-0.306526353649786

Layer 2 -->
  (0)sig:0 weights:[0.488210594906847, 0.488210594906847] bias:-0.0607246554943734


[0.351258218621797]
[0]
[1.06253991967795]
dnjstlr555@owonsig-ui-MacBook-Pro ~ % /usr/local/bin/python /Users/dnjstlr555/Documents/ml/study/fastper.py
Sys -->
Layer 0 -->
  (0)sig:0.9325268695618681 weights:[] bias:0
  (1)sig:0.9918863815054654 weights:[None] bias:0

Layer 1 -->
  (0)sig:1.19246200204395 weights:[0.748630995602632, 0.735557208762477] bias:-0.292613162352982
  (1)sig:1.19246200204395 weights:[0.748630995602632, 0.735557208762477] bias:-0.292613162352982

Layer 2 -->
  (0)sig:1.19822573672802 weights:[0.459839300761331, 0.459839300761331] bias:-0.0508492705778524


[0.362528731575700]
[0]
[1.04501679766917]
dnjstlr555@owonsig-ui-MacBook-Pro ~ % /usr/local/bin/python /Users/dnjstlr555/Documents/ml/study/fastper.py
Sys -->
Layer 0 -->
  (0)sig:0.9391897383914356 weights:[] bias:0
  (1)sig:0.8884312720132258 weights:[None] bias:0

Layer 1 -->
  (0)sig:1.02589749162261 weights:[0.722910972523513, 0.744312560643349] bias:-0.309522867281050
  (1)sig:1.02589749162261 weights:[0.722910972523513, 0.744312560643349] bias:-0.309522867281050

Layer 2 -->
  (0)sig:0.981945593436409 weights:[0.501395857034579, 0.501395857034579] bias:-0.0356043721424605


[0.389668462106676]
[0]
[1.12532826298018]
dnjstlr555@owonsig-ui-MacBook-Pro ~ % /usr/local/bin/python /Users/dnjstlr555/Documents/ml/study/fastper.py
Sys -->
Layer 0 -->
  (0)sig:0.9167205116081598 weights:[] bias:0
  (1)sig:0.7449264600410017 weights:[None] bias:0

Layer 1 -->
  (0)sig:0.504119256155953 weights:[0.591805369834097, 0.592614603059676] bias:-0.479855163676664
  (1)sig:0.504119256155953 weights:[0.591805369834097, 0.592614603059676] bias:-0.479855163676664

Layer 2 -->
  (0)sig:0 weights:[0.184419972893772, 0.184419972893772] bias:-0.479855163676664


[0]
[0]
[0]
```

### 더 해야할점
가끔 모든 값이 0,0,0으로 잘못 학습되어 나오는 경우가 있는데 원인 분석하기<br>
내가 구한 식이 맞을까? 

# 3
학습 마저 고치기
## 출력층 함수 바꾸기
출력층의 함수를 Relu 대신 sigmoid로 바꾸니 엄청난 안정성이 생겼습니다.<br>
랜덤하게 가끔 제대로 학습이 안되는 경우도 있었지만 그래도 전보다는 훨씬 안정된 모습이 나왔습니다<br>
Relu의 경우 학습이 잘되면 매우 빠르게 수렴하지만 안정성이 낮았고 Sigmoid는 점진적으로 수렴하지만 안정성이 높은것 같습니다.<br>
```python
print(sys.fastfeed([0.5,0.5]))
print(sys.fastfeed([0.1,0.1]))
print(sys.fastfeed([0.9,0.9]))
print(sys.fastfeed([0,1]))
print(sys.fastfeed([33,33]))
print(sys.ToString())
```
```
rain for 0 -> 466 / 1 -> 515
evaluates for [1,1] 0.000414060539817004
evaluates for [0.1,0.1] 0.00843978577903156
train for 0 -> 470 / 1 -> 521
evaluates for [1,1] 0.000405458909588757
evaluates for [0.1,0.1] 0.00829880091654394
[0.519587425645496]
[0.0902662710775888]
[0.957704624814341]
[0.411715269721999]
[1.00000000000000]
Sys -->
Layer 0 -->
  (0)sig:33 weights:[] bias:0
  (1)sig:33 weights:[None] bias:0

Layer 1 -->
  (0)sig:92.8357436752349 weights:[1.53404674651969, 1.29512014783160] bias:-0.526763838357612
  (1)sig:35.0894021467293 weights:[0.523500500881911, 0.545772129398353] bias:-0.196594652519408
  (2)sig:51.3123965491348 weights:[0.915439469168330, 0.648342979139997] bias:-0.292424245040025
  (3)sig:0 weights:[-0.132963220792127, 0.0952794798116780] bias:0.728081619260251

Layer 2 -->
  (0)sig:1.00000000000000 weights:[-1.91900344659907, -0.599164971956029, -0.963092336634339, 0.736683216914006] bias:1.77679866970300
```
# 4
현재 학습은 출력층의 뉴런 개수가 1개일때만 제대로 작동함.
## 학습 방법 고치기
```
train for 0 -> 492 / 1 -> 489
evaluates for [1,1] 0.506095656357142
evaluates for [0.1,0.1] 0.0429059494634863
train for 0 -> 498 / 1 -> 493
evaluates for [1,1] 0.506535316633274
evaluates for [0.1,0.1] 0.0407663545388226
[0.565476206260165, 0.0213041669706672] ([0.5,0.5])
[0.238739204886161, 0.158851891278846] ([0.1,0.1])
[0.843755314536744, 0.00250279573365795] ([0.9,0.9])
[0.572570953721872, 0.0209073311553977] [0,1]
[1.00000000000000, 1.26092720310890e-78] [33,33]
```
출력층의 뉴런 갯수를 2개로 늘렸는데 한쪽만 잘 학습이 되고 오른쪽 출력층은 아예 학습이 안되는 모습을 보였습니다.
## 검증
```python
def back(self, **kwargs):
  if t is None: #처음의 경우
      temp=2/len(self.layer.children)*(self.sig-predict)*(self.deltaHZ()) #이 식의 결과는 dCo/dZo(n), n번째 출력 결과에 대한 비용함수의 미분
  else:
      temp=t*self.deltaHZ()  
  for i, n in enumerate(self.layer.ProvidePrevNeu()):
      provid=temp*self.w[i]
      self.w[i]-=temp*n.sig*0.1
      n.back(predict=predict, t=provid)
  self.b-=temp*0.1
```
```python
temp=t*self.deltaHZ()
```
전체 비용함수에 대한 미분 결과를 구해야 했는데, 식 검증 과정에서 무언가 놓친게 있는것 같음<br>
그래서 구해보았는데 층을 나타내는 숫자 위치의 표기를 잘못 표기했음을 다 쓰고 나서 알게되었습니다 ...<br>
h1은 출력층 가장 가까이 있는 은닉층의 첫번째 놈이고 h^2_1 (상단에 2, 아래에 1)은 출력층에서 2번째로 먼 은닉층의 첫번째 놈을 얘기합니다.<br>
Co -> O1, O2, O3... -> h1, h2, h3 ... -> h^2_1, h^2_2 ... 
### 최하위 은닉층
![render](https://user-images.githubusercontent.com/21963949/98175743-87181a00-1f3a-11eb-99b5-bde6bde21a30.png)<br>
위 식의 결과는 첫번째 은닉층에 대한 비용함수의 미분 결과를 나타냅니다 <br>
여기까지는 별 무리가 없지만 그 다음 더 깊은 은닉층을 구할때 식 자체는 같은데.. 결과가 다를것 같은 놈이 생깁니다<br>
![render-3](https://user-images.githubusercontent.com/21963949/98175726-7f587580-1f3a-11eb-84b7-8cd042906837.png)<br>
Z 함수는 필터를 거치기 전 이전 레이어에서의 총 계산 결과입니다.<br>
![render-2](https://user-images.githubusercontent.com/21963949/98175737-841d2980-1f3a-11eb-8493-0b3aaa4f4a1b.png)<br>
이 식들의 총 계산 결과는 dCo/d해당 은닉층을 나타내지만 안의 내용물이 다른데 당연히 계산 결과가 다를것 같았습니다.<br>
그러면 여기서도 비용함수처럼 항을 나눠서 계산을 해야되는것 같은데... 잘 안와닿았습니다 <br>
![img-a5905af106576669](https://user-images.githubusercontent.com/21963949/98206628-8f457900-1f7d-11eb-9e71-bc07efc90177.jpg)<br>
딱봐도 이 식을 적용하는것 같고 h1과 h2는 h^2_1을 가지고 있는게 확실하기 때문에 이렇게 하는게 맞을것 같지만 너무 찝찝해서 한번 짚어보려구 합니다<br>
### 다변수 함수의 미분
![img-e842610b37aca997](https://user-images.githubusercontent.com/21963949/98206611-8c4a8880-1f7d-11eb-8c6a-09a5776fca1d.jpg)일때 연쇄 법칙과 편미분을 사용해서 dz/dt를 구할수 있습니다. [출처](https://suhak.tistory.com/909)<br>
![img-34b9c84b1418b130](https://user-images.githubusercontent.com/21963949/98206610-8bb1f200-1f7d-11eb-8203-fbb676e441bd.jpg)<br>
이런 느낌으로 비용함수에 대한 h1의 기울기에 접근해 보려고 합니다.<br>
### 최하위 은닉층의 기울기
![render](https://user-images.githubusercontent.com/21963949/98175743-87181a00-1f3a-11eb-99b5-bde6bde21a30.png)<br>
먼저 당연하게 받아들일수 있었던 이 식을 먼저 확인해보았습니다. <br>
![img-a0e2d089ee39e454](https://user-images.githubusercontent.com/21963949/98206626-8f457900-1f7d-11eb-8a8a-901273e9802d.jpg)<br>
비용함수가 O1, O2, O3에 대한 식으로 이루어져 있고 그 식들은 h1을 담고 있어서 이런 결과가 나온게 확실히 보입니다.<br>
이렇게 해서 은닉층 갯수만큼 편미분의 결과를 얻을수 있습니다.<br>
![img-bbb51acca0f7b385](https://user-images.githubusercontent.com/21963949/98206624-8eace280-1f7d-11eb-8260-a080dbdc2812.jpg)<br>
즉 이 뜻은 비용함수를 까고 까고 깠을때 나머지를 상수 취급하면<br>
![img-0a5689d1c045a59d](https://user-images.githubusercontent.com/21963949/98206623-8e144c00-1f7d-11eb-9b27-baaa1c182ddd.jpg)<br>
처럼 h1, h2, h3의 문자들로만 나타내지는 다변수 함수가 됩니다.<br> 
결국 그럼 h1, h2, h3는 하위 은닉층인 h^2_1에 영향을 받기 때문에 다음과 같이 정리됩니다.<br>
![img-33114dd84ab71333](https://user-images.githubusercontent.com/21963949/98206621-8e144c00-1f7d-11eb-93b9-fa2f203cf517.jpg)<br>
이때 f는 Co함수의 동작을 가르킵니다. 그러면 h^2_1에 대한 편미분이 가능해졌습니다.<br>
![img-632e2828905f199f](https://user-images.githubusercontent.com/21963949/98206619-8d7bb580-1f7d-11eb-8eb8-ab784dc2f77e.jpg)<br>
![img-5a288f2636b6ac24](https://user-images.githubusercontent.com/21963949/98206613-8ce31f00-1f7d-11eb-89f9-71a7abbbb319.jpg)<br>
비용함수에 대한 h1, h2, h3들의 미분값은 미리 계산이 되어있고 h1/h^2_1, h2/h^2_1 등은 쉽게 구할수 있습니다. <br>이렇게 은닉층 뒤 은닉층의 비용함수 기울기를 검증할 수 있었습니다.<br>그 다음 은닉층의 경우 같은 방법으로 구하면 될거 같습니다.<br>
![img-bad6e218809b8817](https://user-images.githubusercontent.com/21963949/98206607-894f9800-1f7d-11eb-8ac1-b11c0311eef1.jpg)<br>
## 알고리즘 확인
식에 따르면 이전 레이어의 모든 연산이 완료되고 정보가 제공되어야 하지만 분배 법칙에 의해 따로따로 적용시켜도 괜찮을것 같아서 확인해 보았습니다.<br>
경사하강법에 의해 가중치를 업데이트 하는 식은 아래와 같습니다.<br>
![img-bad6e218809b8817](https://user-images.githubusercontent.com/21963949/98356230-9b593580-2066-11eb-8b1f-7845db2a5b97.jpg)
<br>

h^2_1의 h^3_1에 대한 가중치 업데이트 식은 이러합니다.<br>
![img-75756add0e1ddbed](https://user-images.githubusercontent.com/21963949/98356227-9ac09f00-2066-11eb-867b-e10448a314bc.jpg)<br>
이때 알파는 학습률을 나타내는 상수이고 (w)h2:1->h3:1은 h3:1에서 받아오는 출력값에 곱해주는 가중치를 나타냅니다.<br>
이걸 풀어보면 <br>
![img-a38e65818a04aa87](https://user-images.githubusercontent.com/21963949/98356222-9ac09f00-2066-11eb-9455-a51c4a576630.jpg)<br>
이런 결과가 나오는데, (1)번 애들은 바로 이전단계 정보만 잇으면 구할수 있습니다.<br>
비용함수에 대한 미분의 경우 방금 검증한 내용을 토대로<br>
![img-e4186dfb52dc0377](https://user-images.githubusercontent.com/21963949/98356216-98f6db80-2066-11eb-96c4-798cd2c26187.png)<br>
이렇게 정리할 수 있습니다.<br>
다른 층에 대해서도 정리를 해보면 <br>
![img-d6b9de2728c0fd0f](https://user-images.githubusercontent.com/21963949/98356221-9a280880-2066-11eb-8cc9-3cfa39051e43.png)<br>
같은 결과가 나옵니다.<br>
1번식의 경우 출력층에서 일어나는 연산으로써 h1이 사용할 기울기를 제공하는데 사용됩니다. <br>
2번식은 바로 앞의 층의 모든 계산 결과를 사용합니다. 3번식은 더 깊은 층의 기울기인데 비슷한 형태의 계산이 반복되서 더해집니다. 
## 재귀함수
재귀함수 특성상 구할수 있는 정보가 자신의 정보만으로 제한돼 있습니다. 다만 다음 노드에 대한 재귀함수를 호출할때 거기다 정보를 담을 수 있습니다.<br>
그리고 O1->O2->O3.. 순이 아니라 O1->h1->h^2_1->h^2_2... 같이 뿌리내리는 모양으로 호출이 됩니다. 이 점을 염두해 두고 한번 확인해 보겠습니다.<br>
시그마 형태로 나타내면<br>
![img-a59b7cd88f7fb34f](https://user-images.githubusercontent.com/21963949/98356210-95fbeb00-2066-11eb-91f7-46ba162d5149.png)<br>
이렇게 나타낼 수 있습니다. 이 식들은 전전식의 항의 형태를 알려줍니다.<br>
2번 식에서 오른쪽 계수는 이전 노드들의 정보가 있어야 하구 왼쪽 계수는 더 거슬러 올라가 비용함수 정보를 필요로 합니다.<br>
3번 식은 2번식에 비슷한 형태로 오른쪽 계수가 추가되었습니다.<br>
재귀함수는 똑같은 노드에 대해 다시 실행되지 않고 식에서 필요로 하는 모든 정보는 다 한번씩 가지를 내려오면서 얻을 수 있는 정보입니다. 
## 코드 작성
파이썬 느낌의 의사 코드로 알고리즘을 작성해 보겠습니다.
```python
def back(self, t):
  if self.layer is last:
    dt=(dCo/dself)
  else:
    dt=t
  self.b=self.b - dt*(dself/d{self.b})
  for neuron in nextLayer.neurons:
    self.w[neuron]=self.w[neuron] - dt*(dself/d{self.w[neuron]})
    provide=dt*(dself/d{neuron})
    neuron.back(provide)
```
아래는 진짜 코드입니다.
```python
def ActDiff(self):
  return sympy.diff(self.f)
def Z(self):
  prev=self.layer.ProvidePrevSig()
  sum=0
  for i, p in enumerate(prev):
    sum+=self.w[i]*p
  return sum+self.b
def diffSigZ(self):
  return (self.ActDiff().subs(x,self.Z()))
def back(self, **kwargs):
  if self.layer.index<=1:
      return
  t=kwargs.get("t")
  train=kwargs.get("trainData")
  if train==None:
      return -1 
  if t==None:
      dt=(2/len(self.layer.children))*(self.sig-train)
  else:
      dt=t
  for i, neuron in enumerate(self.layer.ProvidePrevNeu()):
      provide=dt*self.diffSigZ()*self.w[i]
      neuron.back(t=provide, trainData=train)
      self.w[i]=self.w[i] - dt*self.diffSigZ()*neuron.sig*0.1
  self.b=self.b - dt*self.diffSigZ()*0.1
```
## 원본 식과의 차이
먼저 값을 제공하는 함수들의 차이점을 확인해볼게요
```python
def ProvideZ(self): #기존
  prevSig=self.layer.ProvidePrevSig()
  providz=0
  for i, sig in enumerate(prevSig):
      if self.w[i] is None:
          self.w[i]=random.uniform(0.5,1)
      providz+=self.w[i]*sig
  return providz+self.b
```
```python
def Z(self):
  prev=self.layer.ProvidePrevSig()
  sum=0
  for i, p in enumerate(prev):
      if self.w[i]==None:
          self.w[i]=random.uniform(0.5,1)
      sum+=self.w[i]*p
  return sum+self.b
```
크게 다른게 없습니다.

```python
def deltaHZ(self): #기존
  return sympy.diff(self.f).subs(x,self.ProvideZ())
```
```python
def ActDiff(self):
  return sympy.diff(self.f)
def diffSigZ(self):
  return (self.ActDiff().subs(x,self.Z()))
```
비슷해 보이는 부분이지만 제일 불안한 부분인데 <br>
sympy 문법이나 사용법이 익숙하지 않아서 이부분을 조금만 건들면 오류가 났었기 때문입니다.<br>
둘의 차이가 있을까요?<br>


```python
if t is None:
  temp=2/len(self.layer.children)*(self.sig-predict)*(self.deltaHZ())
else:
  temp=t*self.deltaHZ()  
for i, n in enumerate(self.layer.ProvidePrevNeu()):
  provid=temp*self.w[i]
  self.w[i]-=temp*n.sig*0.1
  n.back(predict=predict, t=provid)
self.b-=temp*0.1
```
```python
if t==None:
  dt=(2/len(self.layer.children))*(self.sig-train)
else:
  dt=t
for i, neuron in enumerate(self.layer.ProvidePrevNeu()):
  provide=dt*self.diffSigZ()*self.w[i]
  neuron.back(t=provide, trainData=train)
  self.w[i]=self.w[i] - dt*self.diffSigZ()*neuron.sig*0.1
self.b=self.b - dt*self.diffSigZ()*0.1
```
사실 별다른게 없습니다. 뭐가 문제였었을까요??<br>
다만 w[i]의 적용 순서를 다르게 했습니다. 바뀐 w[i]를 보내주는것보다 기존 w[i]를 보내주는게 맞을것 같은데 한번 확인해보겠습니다.<br>

### 순서?
먼저 w[i]를 수정한 모델의 결과입니다.
```
train for 0 -> 0 / 1 -> 1
evaluates for [1,1] 0.972278245495447
evaluates for [0.1,0.1] 0.155636759156753
train for 0 -> 5 / 1 -> 6
evaluates for [1,1] 0.971560653710321
evaluates for [0.1,0.1] 0.150932099111381
train for 0 -> 10 / 1 -> 11
evaluates for [1,1] 0.970648548890313

...

train for 0 -> 491 / 1 -> 480
evaluates for [1,1] 0.00750842673013943
evaluates for [0.1,0.1] 0.0328476299688556
train for 0 -> 496 / 1 -> 485
evaluates for [1,1] 0.00752206990583356
evaluates for [0.1,0.1] 0.0320794762073455
train for 0 -> 504 / 1 -> 487
evaluates for [1,1] 0.00811173287592027
evaluates for [0.1,0.1] 0.0308144292736076
[0.548688837876099, 0.543857771713573] ([0.5,0.5] 결과)
[0.184311763879187, 0.166172003929709] ([0.1,0.1] 결과)
[0.867398768673058, 0.877048240556812] ([0.9,0.9] 결과)
[0.525312161749080, 0.525961070164471] ([0,1] 결과)
[1.00000000000000, 1.00000000000000] ([33,33])
```
그다음 w[i]를 보내고 수정하는 경우입니다
```
train for 0 -> 1 / 1 -> 0
evaluates for [1,1] 0.968304652942361
evaluates for [0.1,0.1] 0.157708162416945
train for 0 -> 3 / 1 -> 8
evaluates for [1,1] 0.963031348936692
evaluates for [0.1,0.1] 0.161782493368296
train for 0 -> 6 / 1 -> 15
evaluates for [1,1] 0.958688433524327
evaluates for [0.1,0.1] 0.162906104887089

...

evaluates for [0.1,0.1] 0.0327027136203697
train for 0 -> 492 / 1 -> 489
evaluates for [1,1] 0.00681448236954570
evaluates for [0.1,0.1] 0.0319198793221648
train for 0 -> 500 / 1 -> 491
evaluates for [1,1] 0.00746384888933472
evaluates for [0.1,0.1] 0.0304091667170774
[0.536795512134298, 0.536039309218853] ([0.5,0.5] 결과)
[0.173210083952417, 0.167586743191813] ([0.1,0.1] 결과)
[0.865056965236271, 0.868942781113436] ([0.9,0.9] 결과)
[0.507855021162558, 0.499870887000164] ([0,1] 결과)
[1.00000000000000, 1.00000000000000] ([33,33])
```
둘의 의미있는 차이점은 보이지 않았습니다. 어짜피 경사하강법의 개념을 생각해 보면 그 값의 크기보다 부호가 더 중요하기 때문인것으로 생각됩니다.<br>
## 결론
출력층이 한개일때만 작동하는 오류를 수정했습니다. <br>
그리고 편미분의 기호와 일반 델타 기호의 차이점을 잘 모르고 처음엔 막썼는데(검증의 최하위 은닉층 파트쪽) 계산하다 보니까 확실히 알게 되었습니다. <br>
# 5
유전 알고리즘 결합하기
## 시스템 정비하기
```python
import mlp
model=mlp.nsystem(inputSize=2)
model.StackLayer(number=4, function=mlp.ReLU) 
model.StackLayer(number=1, function=mlp.Sigmoid)
x=[[0.03, 0.01], [0.9, 0.97]]
y=[[0], [1]]
for i, data in enumerate(x):
  model.train(inputData=data, result=y[i], trainMethod=mlp.Method.Genetic, poolSize=10, muchance=0.1)
  model.train(inputData=data, result=y[i], trainMethod=mlp.Method.GradientDescent, alpha=0.05)
```
## 유전 알고리즘
유전 알고리즘의 가장 큰 장점은 미분 과정이 필요했던 경사 하강법과는 다르게 연산 과정이 별로 없다는것입니다. <br>
자연 선택의 원리를 따라 적합도가 높은 순서대로 유전자를 합치고 돌연변이 시키고 하는 과정을 계속 반복하면 환경에 적합한 유전자를 찾을 수 있습니다.<br>
### 만들기
두 리스트간 교차 알고리즘입니다.
```python
@staticmethod
def cross(x :list, y :list, muchance :float, sigma :float) -> list:
  if len(x)!=len(y):
      print(f"Length does not match -> {len(x)} {len(y)}")
      raise
  l=len(x)
  dvid=random.randint(0, l)
  result=[*x[:dvid], *y[dvid:]]
  for i, a in enumerate(result):
      if random.uniform(0, 1) < muchance:
          result[i]+=random.gauss(0,sigma)
  return result
```
상수들을 유전자로 나타내고 처리하는 알고리즘입니다.
```python
size=3 #세대당 모델 개수
muchance=0.01 #돌연변이 확률
sigma=1 #돌연변이시 표준편차
memory=sys.memory #각 세대의 모델들이 담겨있는 장소
if not memory: #초기 설정시
    for _ in itertools.repeat(None, size):
        memory.append({"model":sys.Copy(), "fitness":0}) #sys.Copy()는 스스로를 복제함
    memory.append({"proto":sys.Copy()})
proto=[i["proto"] for i in memory if "proto" in i][0] #더미 모델 불러오기
parentPool=[i for i in memory if "model" in i] 
maxFit=0
maxModel=proto
for m in parentPool:
    model=m.get("model")
    if not model:
        continue
    fit=model.evaluate(inputData, result)
    m.update({"fitness":fit}) #적합도 찾기
    if maxFit<fit:
        maxFit=fit
        maxModel=model #적합도가 최대인 모델 불러오기
if maxModel is not proto:
    sys.ImportModel(maxModel)
fTable=[(0 if f.get("fitness") is None else f.get("fitness")) for f in parentPool] #적합도만 불러오기
p=cls.softmax(fTable) #확률 적용을 위해 0~1 사이의 값으로 나타내기
childPool=[]
for _ in itertools.repeat(None, math.floor(3*size/4)):
    parent=np.random.choice(parentPool, p=p, size=2, replace=True)
    w=[]
    b=[]
    for k in parent:
        (tw, tb)=cls.GeneticGetTable(k) #해당 모델의 w와 b를 리스트 형ㅌ태로 불러오기
        if tw is not None and tb is not None:
            w.append(tw)
            b.append(tb)
    childW=cls.cross(w[0], w[1], muchance, sigma) #걔네끼리 교차시키기
    childB=cls.cross(b[0], b[1], muchance, sigma)
    child=proto.Copy()
    child.SetParams(childW, childB) #프로토를 복사하고 파라미터 적용하기
    childPool.append({"model":child, "fitness":0}) 
for _ in itertools.repeat(None, math.floor(size/4)):
    child=proto.Copy()
    childPool.append({"model":child, "fitness":0}) #나머지는 랜덤 애들로 채우기 
childPool.append({"proto":proto})
sys.memory=childPool
```
문제는 너무나도 직접 설정해줘야됄 값이 많아져서 적합한 파라미터를 찾는데 시간이 걸릴것 같습니다.
```
train for 0 -> 1 / 1 -> 0 / 10
evaluates for [1,1] 0.872286078486012
evaluates for [0.1,0.1] 0.0565342490624910
train for 0 -> 6 / 1 -> 5 / 10

...

train for 0 -> 170 / 1 -> 191 / 10
evaluates for [1,1] 0.859670959506938
evaluates for [0.1,0.1] 0.0571793510040143
train for 0 -> 177 / 1 -> 194 / 10
evaluates for [1,1] 0.851449563877306
evaluates for [0.1,0.1] 0.0579993652263276
```
# 6
그러다 아주 중대한 실수를 깨달았습니다. fitness로 저장되는 evaluate 함수는 잘 작동할수록 0에 가까워지는데 그러면 오히려 선택될 확률이 적어집니다.
```python
fit=abs(model.evaluate(inputData, result)-1)
```
```
train for 0 -> 189 / 1 -> 222 / 10
evaluates for [1,1] 0.996959111224677
evaluates for [0.1,0.1] 0.000765316594234806

...

train for 0 -> 225 / 1 -> 256 / 10
evaluates for [1,1] 0.996577768254929
evaluates for [0.1,0.1] 0.00100605509621974
train for 0 -> 228 / 1 -> 263 / 10
evaluates for [1,1] 0.995173180277689
evaluates for [0.1,0.1] 0.00149081716889370
```
그런데 오히려 더 나빠졌습니다. 좀더 결과를 찾아본 결과... 새로운 개체를 반환해야 했던 Copy 함수에 문제가 생겼음을 알게되었습니다.<br>
리스트를 복사하지 않고 그 리스트 변수의 주소를 복사하는 방식이라는 것을 검색 뒤에 알게 되었습니다.<br>
다만 이 문제를 그냥 해결하기 위해서는 deepcopy라는것을 사용해야 되는데 얘는 너무 세부적인것까지 복사해서 퍼포먼스에 문제가 생기게 되었습니다. 직접 복사 함수를 만들었습니다.<br>

```python
class nsystem:
  def Copy(self):
    p=self.__class__()
    p.ImportProto(self.Proto())
    return p
  def Proto(self):
    t=[]
    for l in self.children:
        t.append(l.Proto())
    return [t, len(self.children)]
class nlayer:
  def Proto(self):
    sender=[]
    for n in self.children:
        (index, f, w, b, sig)=n.Proto()
        sender.append([index, f, w, b, sig])
    return [sender, self.index, len(self.children)]
class nneuron:
  def Proto(self):
    return (self.index, self.f, self.w, self.b, self.sig)

...

sys=mlp.nsystem(inputSize=2)
sys.StackLayer(number=3, function=mlp.ReLU)
new=sys.Copy()
new.ImportModel(sys)
print(f"ImportModel Works:{sys.ToString()==new.ToString()}") # True

```
계속 유전 알고리즘이 작동 안됬던 이유는 복사한답시고 엉뚱한 모델을 만들었기 때문이었습니다.<br>
추가적으로 우수 인자를 살리는 비율, 랜덤 인자를 만드는 비율등의 파라미터를 추가하고 학습을 진행했습니다. <br>

```python
model=mlp.nsystem(inputSize=2)
model.StackLayer(number=3, function=mlp.ReLU)
model.StackLayer(number=1, function=mlp.Sigmoid)


for i in range(500):
    rand = random.choice((0,1))
    if rand==0:
        model.train([random.uniform(0.7,1),random.uniform(0.7,1)], [1], trainMethod=mlp.Method.Genetic, poolSize=10, muChance=0.05, eliteRatio=0.4, randRatio=0.05)
    else:
        model.train([random.uniform(0,0.3),random.uniform(0,0.3)], [0], trainMethod=mlp.Method.Genetic, poolSize=10, muChance=0.05, eliteRatio=0.2, randRatio=0.05)
    if i%10==0:
        print(f"{model.evaluate([0,0], [0])} {model.evaluate([0.99,0.99], [1])}")

print(f"{model.fastfeed([0,0])} {model.fastfeed([1,1])}")

```
```
0.00215963096408225 0.998134927377352
0.0000120312150149738 0.999872409190476

...

9.74649021744538E-13 2.67017438916898E-14
5.95818765302186E-13 1.09122041716011E-14
1.08920995835101E-11 7.42510829654135E-18
7.25258549952650E-10 9.70479399094525E-18
0.0387770371055772 4.33314561650588E-19
[1.96815132889355e-10] [0.999999652584768] (0,0) (1,1)
```
으 드디어 되었습니다. 다만 가끔 값이 튀어서 이상한 쪽으로 학습되기도 합니다.<br>

# 7
실제 활용
## 숙제를 해올 확률
kaggle 검색중에 부모 학력, 시험점수와 숙제를 해온 여부를 정리한 파일이 있어서 한번 시험해볼려고 합니다.<br>
```python
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
        return [self.gender, self.race, self.pedu, self.lunch, self.math, self.read, self.write]
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
model=mlp.nsystem(inputSize=7)
model.StackLayer(number=8, function=mlp.ReLU)
model.StackLayer(number=8, function=mlp.ReLU)
model.StackLayer(number=1, function=mlp.Sigmoid)

random.shuffle(dataset)
for i, d in enumerate(dataset):
    model.train(d.ToListOnlyInfo(), [d.IsTestOkay()], trainMethod=mlp.Method.GradientDescent, alpha=0.001)
f = open('store.pckl', 'wb')
pro=model.Proto()
pickle.dump(pro, f)
f.close()
```
학습을 이렇게 처리하고 제 수학과 국어, 평소 점심등을 반영한... 데이터를 모델에 집어넣었습니다.
```python
f = open('store.pckl', 'rb')
obj = pickle.load(f)
f.close()

model=mlp.nsystem()
model.ImportProto(obj)
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
```

```
[8.14014430739655e-2913]
[7.09567335610614e-2663]
[2.04331036154889e-2748]
[2.11505698718984e-2428]
[2.12103864605446e-3612]
```
현재 어떤 값을 집어넣든 거의 0에 가까운 숫자가 나왔습니다. 테스트 하는 학생의 비율이 4:6정도 되어서 그래도 어느정도는 1에 가까울줄 알았는데 문제가 있는것 같아요