# pymlp

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
  if self.layer.index<=0: return 0
  temp=0
  t=None
  predict=[]
  for key, value in kwargs.items():
      if key=="predict": 
          predict=value
      elif key=="t": 
          t=value #이전 레이어의 뉴런에서 받은 값
  if predict is None: return -1 
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
전체 비용함수에 대한 미분 결과를 구해야 했는데, 식 검증 과정에서 무언가 놓친게 있는것 같았습니다. <br>
식을 계산해 보면서 생긴 가장 큰 문제는 깊은 은닉층의 계산이었습니다. <br>
![render](https://user-images.githubusercontent.com/21963949/98175743-87181a00-1f3a-11eb-99b5-bde6bde21a30.png)<br>
위 식의 결과는 첫번째 은닉층에 대한 비용함수의 미분 결과를 나타냅니다 <br>
여기까지는 별 무리가 없지만 그 다음 더 깊은 은닉층을 구할때 식 자체는 같은데.. 결과가 다를것 같은 놈이 생깁니다<br>
![render-3](https://user-images.githubusercontent.com/21963949/98175726-7f587580-1f3a-11eb-84b7-8cd042906837.png)<br>
![render-2](https://user-images.githubusercontent.com/21963949/98175737-841d2980-1f3a-11eb-8493-0b3aaa4f4a1b.png)<br>
Z 함수는 필터를 거치기 전 이전 레이어에서의 총 계산 결과입니다.<br>
이 식들의 총 계산 결과는 dCo/d해당 은닉층을 나타내지만 안의 내용물이 다른데 당연히 계산 결과가 다를것 같았습니다.<br>
애초에 아무리 미분 기호가 담긴 식이 분수 형태로 약분이 된다고는 하지만, 이게 
확인을 위해 재귀함수를 호출할때 보내는 값의 의미를 나타내는 문자열을 표시하게 했습니다.<br>
```python
sys=mlp.nsystem()
layer=sys.CreateLayer(4,2) #층이 4개
sys.CreateNeurons(4, layer[1], mlp.ReLU, 1) #1번 레이어는 입력, 2번 레이어는 ReLU
sys.CreateNeurons(4, layer[2], mlp.ReLU, 1) #3번 레이어도 ReLU
sys.CreateNeurons(2, layer[3], mlp.Sigmoid, 1) #마지막 4번은 Sigmoid

sys.feed([0.3,0.3])
sys.Activate()
result=[0.9,0.9]
for i, n in enumerate(sys.out().children):
    n.descback(predict=result[i])
```
```python
def descback(self, **kwargs):
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
```
```
(dCo/dZ3:0) => 0.0227924747036237
(dCo/dZ3:0)(dZ3:0/dH2:0)(dH2:0/dZ2:0) => 0.0166515536496952
(dCo/dZ3:0)(dZ3:0/dH2:0)(dH2:0/dZ2:0)(dZ2:0/dH1:0)(dH1:0/dZ1:0) => 0.0114297530746209
(dCo/dZ3:0)(dZ3:0/dH2:0)(dH2:0/dZ2:0)(dZ2:0/dH1:1)(dH1:1/dZ1:1) => 0.0128708240028593
(dCo/dZ3:0)(dZ3:0/dH2:0)(dH2:0/dZ2:0)(dZ2:0/dH1:2)(dH1:2/dZ1:2) => 0.0121948226256797
(dCo/dZ3:0)(dZ3:0/dH2:0)(dH2:0/dZ2:0)(dZ2:0/dH1:3)(dH1:3/dZ1:3) => 0.0147195158535989
(dCo/dZ3:0)(dZ3:0/dH2:1)(dH2:1/dZ2:1) => 0.0207492358169774
(dCo/dZ3:0)(dZ3:0/dH2:1)(dH2:1/dZ2:1)(dZ2:1/dH1:0)(dH1:0/dZ1:0) => 0.0155725578720509
(dCo/dZ3:0)(dZ3:0/dH2:1)(dH2:1/dZ2:1)(dZ2:1/dH1:1)(dH1:1/dZ1:1) => 0.0105325002288606
(dCo/dZ3:0)(dZ3:0/dH2:1)(dH2:1/dZ2:1)(dZ2:1/dH1:2)(dH1:2/dZ1:2) => 0.0189852180495419
(dCo/dZ3:0)(dZ3:0/dH2:1)(dH2:1/dZ2:1)(dZ2:1/dH1:3)(dH1:3/dZ1:3) => 0.0147849574497214
(dCo/dZ3:0)(dZ3:0/dH2:2)(dH2:2/dZ2:2) => 0.0129690523241637
(dCo/dZ3:0)(dZ3:0/dH2:2)(dH2:2/dZ2:2)(dZ2:2/dH1:0)(dH1:0/dZ1:0) => 0.00903188234300677
(dCo/dZ3:0)(dZ3:0/dH2:2)(dH2:2/dZ2:2)(dZ2:2/dH1:1)(dH1:1/dZ1:1) => 0.0120503450057629
(dCo/dZ3:0)(dZ3:0/dH2:2)(dH2:2/dZ2:2)(dZ2:2/dH1:2)(dH1:2/dZ1:2) => 0.0116958163057677
(dCo/dZ3:0)(dZ3:0/dH2:2)(dH2:2/dZ2:2)(dZ2:2/dH1:3)(dH1:3/dZ1:3) => 0.00801058826769267
(dCo/dZ3:0)(dZ3:0/dH2:3)(dH2:3/dZ2:3) => 0.0153999646687269
(dCo/dZ3:0)(dZ3:0/dH2:3)(dH2:3/dZ2:3)(dZ2:3/dH1:0)(dH1:0/dZ1:0) => 0.00823117352727845
(dCo/dZ3:0)(dZ3:0/dH2:3)(dH2:3/dZ2:3)(dZ2:3/dH1:1)(dH1:1/dZ1:1) => 0.0125298028187108
(dCo/dZ3:0)(dZ3:0/dH2:3)(dH2:3/dZ2:3)(dZ2:3/dH1:2)(dH1:2/dZ1:2) => 0.0134852463814999
(dCo/dZ3:0)(dZ3:0/dH2:3)(dH2:3/dZ2:3)(dZ2:3/dH1:3)(dH1:3/dZ1:3) => 0.00838235072363132
(dCo/dZ3:1) => 0.0128994383705544
(dCo/dZ3:1)(dZ3:1/dH2:0)(dH2:0/dZ2:0) => 0.0124624423188130
(dCo/dZ3:1)(dZ3:1/dH2:0)(dH2:0/dZ2:0)(dZ2:0/dH1:0)(dH1:0/dZ1:0) => 0.00854461781782720
(dCo/dZ3:1)(dZ3:1/dH2:0)(dH2:0/dZ2:0)(dZ2:0/dH1:1)(dH1:1/dZ1:1) => 0.00962637884882033
(dCo/dZ3:1)(dZ3:1/dH2:0)(dH2:0/dZ2:0)(dZ2:0/dH1:2)(dH1:2/dZ1:2) => 0.00911684407466581
(dCo/dZ3:1)(dZ3:1/dH2:0)(dH2:0/dZ2:0)(dZ2:0/dH1:3)(dH1:3/dZ1:3) => 0.0110079273742798
(dCo/dZ3:1)(dZ3:1/dH2:1)(dH2:1/dZ2:1) => 0.0122345996371487
(dCo/dZ3:1)(dZ3:1/dH2:1)(dH2:1/dZ2:1)(dZ2:1/dH1:0)(dH1:0/dZ1:0) => 0.00917035537819763
(dCo/dZ3:1)(dZ3:1/dH2:1)(dH2:1/dZ2:1)(dZ2:1/dH1:1)(dH1:1/dZ1:1) => 0.00620247876434274
(dCo/dZ3:1)(dZ3:1/dH2:1)(dH2:1/dZ2:1)(dZ2:1/dH1:2)(dH1:2/dZ1:2) => 0.0111821457184559
(dCo/dZ3:1)(dZ3:1/dH2:1)(dH2:1/dZ2:1)(dZ2:1/dH1:3)(dH1:3/dZ1:3) => 0.00870738296060343
(dCo/dZ3:1)(dZ3:1/dH2:2)(dH2:2/dZ2:2) => 0.00982443362228274
(dCo/dZ3:1)(dZ3:1/dH2:2)(dH2:2/dZ2:2)(dZ2:2/dH1:0)(dH1:0/dZ1:0) => 0.00683595902156630
(dCo/dZ3:1)(dZ3:1/dH2:2)(dH2:2/dZ2:2)(dZ2:2/dH1:1)(dH1:1/dZ1:1) => 0.00912451358941298
(dCo/dZ3:1)(dZ3:1/dH2:2)(dH2:2/dZ2:2)(dZ2:2/dH1:2)(dH1:2/dZ1:2) => 0.00885373849834112
(dCo/dZ3:1)(dZ3:1/dH2:2)(dH2:2/dZ2:2)(dZ2:2/dH1:3)(dH1:3/dZ1:3) => 0.00606301636967874
(dCo/dZ3:1)(dZ3:1/dH2:3)(dH2:3/dZ2:3) => 0.00879155920609353
(dCo/dZ3:1)(dZ3:1/dH2:3)(dH2:3/dZ2:3)(dZ2:3/dH1:0)(dH1:0/dZ1:0) => 0.00469270020610435
(dCo/dZ3:1)(dZ3:1/dH2:3)(dH2:3/dZ2:3)(dZ2:3/dH1:1)(dH1:1/dZ1:1) => 0.00714881485415029
(dCo/dZ3:1)(dZ3:1/dH2:3)(dH2:3/dZ2:3)(dZ2:3/dH1:2)(dH1:2/dZ1:2) => 0.00769191256171703
(dCo/dZ3:1)(dZ3:1/dH2:3)(dH2:3/dZ2:3)(dZ2:3/dH1:3)(dH1:3/dZ1:3) => 0.00477976667108846
```
이 결과에서 Z와 H 뒤에 붙는 3:1, 2:1 같은 애들은 (레이어의 인덱스):(레이어 내 인덱스)를 가르킵니다.<br> 
3:1의 경우 4번째 레이어의 2번째 뉴런(배열의 index는 0부터 시작하니까), 2:0은 3번째 층의 1번째 뉴런을 얘기합니다.<br>
지금 알아야 할 것은 2번째 층에 있는 뉴런들의 기울기가 제대로 계산이 되는것임. <br>
```
(dCo/dZ3:0) => 0.0227924747036237
```
아까 4개의 층을 생성하기로 설정했으로 dZ3:0은 출력층의 1번째 뉴런을 가르킵니다.<br>
이를 비용함수에 대해 편미분한 결과입니다.
```
(dCo/dZ3:0)(dZ3:0/dH2:0)(dH2:0/dZ2:0) => 0.0166515536496952
(dCo/dZ3:1)(dZ3:1/dH2:0)(dH2:0/dZ2:0) => 0.0124624423188130
```
각각 비용함수를 Z2:0에 대해 편미분한 결과입니다.<br>
이 둘을 합치면 Z2:0에 대한 비용함수의 기울기가 되는것은 알고있는데...

