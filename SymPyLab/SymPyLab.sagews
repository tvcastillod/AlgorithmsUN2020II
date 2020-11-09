︠b4ccffc6-99ba-46da-b4f5-499f4ee07edc︠
%md 
# SymPyLab

SymPy’s documentation
- https://docs.sympy.org/latest/index.html

## SymPy’s polynomials 
- https://docs.sympy.org/latest/modules/polys/basics.html#polynomials 

- (x-1)(x-2)(x-3)(x-4)(x-5)(x-6)(x-7)(x-8)(x-9)(x-10)

<img src="https://raw.githubusercontent.com/tvcastillod/AlgorithmsUN2020II/master/SymPyLab/sympylabwolfram1.png" />
︡69d46f1e-5123-44d9-89bb-3f1876c3908b︡{"done":true,"md":"# SymPyLab\n\nSymPy’s documentation\n- https://docs.sympy.org/latest/index.html\n\n## SymPy’s polynomials \n- https://docs.sympy.org/latest/modules/polys/basics.html#polynomials \n\n- (x-1)(x-2)(x-3)(x-4)(x-5)(x-6)(x-7)(x-8)(x-9)(x-10)\n\n<img src=\"https://raw.githubusercontent.com/tvcastillod/AlgorithmsUN2020II/master/SymPyLab/sympylabwolfram1.png\" />"}
︠7dcbf8fd-040b-4479-b704-7fd72302447cs︠
from sympy import Symbol
from sympy import div

x = Symbol('x')

p = (x-1)*(x-2)*(x-3)*(x-4)*(x-5)*(x-6)*(x-7)*(x-8)*(x-9)*(x-10)

p, r = div(p,  x-1)

print(p)
print(r)

p, r = div(p,  x-2)

print(p)
print(r)

p, r = div(p,  x-3)

print(p)
print(r)

p, r = div(p,  x-4)

print(p)
print(r)
︡629734c7-63cb-4bb7-8dae-3c58c0ec05fe︡{"stdout":"x**9 - 54*x**8 + 1266*x**7 - 16884*x**6 + 140889*x**5 - 761166*x**4 + 2655764*x**3 - 5753736*x**2 + 6999840*x - 3628800\n"}︡{"stdout":"0\n"}︡{"stdout":"x**8 - 52*x**7 + 1162*x**6 - 14560*x**5 + 111769*x**4 - 537628*x**3 + 1580508*x**2 - 2592720*x + 1814400\n"}︡{"stdout":"0\n"}︡{"stdout":"x**7 - 49*x**6 + 1015*x**5 - 11515*x**4 + 77224*x**3 - 305956*x**2 + 662640*x - 604800\n"}︡{"stdout":"0\n"}︡{"stdout":"x**6 - 45*x**5 + 835*x**4 - 8175*x**3 + 44524*x**2 - 127860*x + 151200\n"}︡{"stdout":"0\n"}︡{"done":true}
︠beb0c0da-374b-4383-9c04-5f8adcd35e81s︠
%md <img src="https://raw.githubusercontent.com/tvcastillod/AlgorithmsUN2020II/master/SymPyLab/sympylabwolfram2.png" />
︡3221489e-6416-46ba-8906-30e20d1b18f3︡{"md":"<img src=\"https://raw.githubusercontent.com/tvcastillod/AlgorithmsUN2020II/master/SymPyLab/sympylabwolfram2.png\" />"}︡{"done":true}
︠181a87cc-138a-46c9-b725-4e3042357202s︠
from sympy.plotting import plot as symplot
from sympy import * 
x = Symbol('x') 
symplot((x-1)*(x-2)*(x-3)*(x-4)*(x-5)*(x-6)*(x-7)*(x-8)*(x-9)*(x-10), ylim = (-50000,50000), xlim = (0,11),line_color='red', title = "polinomio original")
symplot(((x-1)*(x-2)*(x-3)*(x-4)*(x-5)*(x-6)*(x-7)*(x-8)*(x-9)*(x-10))/(x-1), ylim = (-50000,50000), xlim = (0,11),line_color='red', title = "polinomio 2")
symplot(((x-1)*(x-2)*(x-3)*(x-4)*(x-5)*(x-6)*(x-7)*(x-8)*(x-9)*(x-10))/(x-2), ylim = (-50000,50000), xlim = (0,11),line_color='red', title = "polinomio 3")
symplot(((x-1)*(x-2)*(x-3)*(x-4)*(x-5)*(x-6)*(x-7)*(x-8)*(x-9)*(x-10))/(x-3), ylim = (-50000,50000), xlim = (0,11),line_color='red', title = "polinomio 4")
symplot(((x-1)*(x-2)*(x-3)*(x-4)*(x-5)*(x-6)*(x-7)*(x-8)*(x-9)*(x-10))/(x-4), ylim = (-50000,50000), xlim = (0,11),line_color='red', title = "polinomio 5")
︡86ffa2ff-9ea1-49fe-973f-a807f46600bd︡{"file":{"filename":"c895d374-607f-4ade-89e4-19ae9159bca5.svg","show":true,"text":null,"uuid":"822fd7a0-e43e-4dad-b964-ee1aba8748db"},"once":false}︡{"stdout":"<sympy.plotting.plot.Plot object at 0x7fa540d8a3d0>"}︡{"stdout":"\n"}︡{"file":{"filename":"eca06e07-92be-4e76-896b-db1129717562.svg","show":true,"text":null,"uuid":"a8f8d97d-8518-49fe-8a46-20846f84feb6"},"once":false}︡{"stdout":"<sympy.plotting.plot.Plot object at 0x7fa53ed67d00>"}︡{"stdout":"\n"}︡{"file":{"filename":"a5c37178-f65c-4340-9ba3-9d09e5390498.svg","show":true,"text":null,"uuid":"b42ac7fb-7f4d-47b5-90a0-73f626d07e31"},"once":false}︡{"stdout":"<sympy.plotting.plot.Plot object at 0x7fa53ecf2df0>"}︡{"stdout":"\n"}︡{"file":{"filename":"76c7e969-4dd8-454c-b88b-f4a682dcacad.svg","show":true,"text":null,"uuid":"2fad0ce0-4bb5-4966-884c-03f0f21645d5"},"once":false}︡{"stdout":"<sympy.plotting.plot.Plot object at 0x7fa53ec10a30>"}︡{"stdout":"\n"}︡{"file":{"filename":"467cd3d2-7a53-41e5-adf4-4df81d573f33.svg","show":true,"text":null,"uuid":"ee6f3ada-d930-416f-b6b3-86cab8e3c186"},"once":false}︡{"stdout":"<sympy.plotting.plot.Plot object at 0x7fa53ed1b700>"}︡{"stdout":"\n"}︡{"done":true}
︠cf6a0870-fa45-4a6e-93de-8054953187b1s︠
%md ## SymPy’s polynomial simple univariate polynomial factorization
- https://docs.sympy.org/latest/modules/polys/wester.html#simple-univariate-polynomial-factorization
- factor(x^10 - 55 x^9 + 1320 x^8 - 18150 x^7 + 157773 x^6 - 902055 x^5 + 3416930 x^4 - 8409500 x^3 + 12753576 x^2 - 10628640 x + 3628800)
<img src="https://raw.githubusercontent.com/tvcastillod/AlgorithmsUN2020II/master/SymPyLab/sympylabwolfram3.png" />
︡1570046a-62e2-4fd1-ab8f-d910d2111775︡{"md":"- https://docs.sympy.org/latest/modules/polys/wester.html#simple-univariate-polynomial-factorization\n- factor(x^10 - 55 x^9 + 1320 x^8 - 18150 x^7 + 157773 x^6 - 902055 x^5 + 3416930 x^4 - 8409500 x^3 + 12753576 x^2 - 10628640 x + 3628800)\n<img src=\"https://raw.githubusercontent.com/tvcastillod/AlgorithmsUN2020II/master/SymPyLab/sympylabwolfram3.png\" />"}︡{"done":true}
︠4aa5def0-65a1-40b3-89a8-517127c1b003s︠
from sympy import *
x = Symbol('x')
factor(x**10 - 55*x**9 + 1320*x**8 - 18150*x**7 + 157773*x**6 - 902055*x**5 + 3416930*x**4 - 8409500*x**3 + 12753576*x**2 - 10628640*x + 3628800)
︡bd352137-93a5-4a8a-a898-22b7a444f836︡{"stdout":"(x - 10)*(x - 9)*(x - 8)*(x - 7)*(x - 6)*(x - 5)*(x - 4)*(x - 3)*(x - 2)*(x - 1)\n"}︡{"done":true}
︠7e6efb17-04dc-48e0-96be-8e6ec51c8927︠
%md 
## SymPy’s solvers
- https://docs.sympy.org/latest/tutorial/solvers.html
- (x\*\*10 - 55\*x\*\*9 + 1320\*x\*\*8 - 18150\*x\*\*7 + 157773\*x\*\*6 - 902055\*x\*\*5 + 3416930\*x\*\*4 - 8409500\*x\*\*3 + 12753576\*x\*\*2 - 10628640\*x + 3628800) = 0

︡ad887fec-2ed5-4f3a-8e46-16a7b3a04e36︡{"done":true,"md":"## SymPy’s solvers\n- https://docs.sympy.org/latest/tutorial/solvers.html\n- (x\\*\\*10 - 55\\*x\\*\\*9 + 1320\\*x\\*\\*8 - 18150\\*x\\*\\*7 + 157773\\*x\\*\\*6 - 902055\\*x\\*\\*5 + 3416930\\*x\\*\\*4 - 8409500\\*x\\*\\*3 + 12753576\\*x\\*\\*2 - 10628640\\*x + 3628800) = 0"}
︠d7a12229-32da-4842-aa6d-ef5c5c83bf8b︠
%md 
<img src="https://raw.githubusercontent.com/tvcastillod/AlgorithmsUN2020II/master/SymPyLab/sympylabwolfram4.png" />
︡bcaeba26-767c-4ad9-aa69-89dab805bc0f︡{"done":true,"md":"<img src=\"https://raw.githubusercontent.com/tvcastillod/AlgorithmsUN2020II/master/SymPyLab/sympylabwolfram4.png\" />"}
︠63cb8b4a-0611-4952-845a-db7f8b71be56s︠
from sympy import *
x = Symbol('x')
solveset(Eq(x**10 - 55*x**9 + 1320*x**8 - 18150*x**7 + 157773*x**6 - 902055*x**5 + 3416930*x**4 - 8409500*x**3 + 12753576*x**2 - 10628640*x + 3628800), x)
︡1e73fea3-8a76-40ec-a271-74a2bf535cf5︡{"stdout":"FiniteSet(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)\n"}︡{"done":true}
︠b347485b-9bb2-487c-83c5-586a68eb5021s︠
%md ## SymPy’s Symbolic and Numerical Complex Evaluations
- https://docs.sympy.org/latest/modules/evalf.html
```
x = -5*x1 + 15*I*x2
y = 3*y1 + 9*I*y2
z = -8*z1 - 4*I*z2
```
︡3dacc3c3-0231-429d-9af7-496fcb56f722︡{"md":"- https://docs.sympy.org/latest/modules/evalf.html\n```\nx = -5*x1 + 15*I*x2\ny = 3*y1 + 9*I*y2\nz = -8*z1 - 4*I*z2\n```"}︡{"done":true}
︠75174b11-feff-43a7-9f28-10f9eae8a398s︠
%md <img src="https://raw.githubusercontent.com/tvcastillod/AlgorithmsUN2020II/master/SymPyLab/sympylabwolfram5.png" />
︡54858207-7f32-46c3-9392-735e9637c677︡{"md":"<img src=\"https://raw.githubusercontent.com/tvcastillod/AlgorithmsUN2020II/master/SymPyLab/sympylabwolfram5.png\" />"}︡{"done":true}
︠e96fea38-a599-40e6-b645-798eef1cb0b6s︠
from sympy import *
x1, x2, y1, y2, z1, z2 = symbols("x1 x2 y1 y2 z1 z2", real=True)  

x = -5*x1 + 15*I*x2
y = 3*y1 + 9*I*y2
z = -8*z1 - 4*I*z2

print(x*y*z)
print(expand(x*y*z))
print(expand((x*y)*z))
print(expand(x*(y*z)))

w = N(1/(sqrt(2) - 17*I), 25)  # aproximación de función N con 20 dígitos
print('w=',w)
︡c5f6471d-2c48-4477-8141-a3fb86d235bd︡{"stdout":"(-5*x1 + 15*I*x2)*(3*y1 + 9*I*y2)*(-8*z1 - 4*I*z2)\n"}︡{"stdout":"120*x1*y1*z1 + 60*I*x1*y1*z2 + 360*I*x1*y2*z1 - 180*x1*y2*z2 - 360*I*x2*y1*z1 + 180*x2*y1*z2 + 1080*x2*y2*z1 + 540*I*x2*y2*z2\n"}︡{"stdout":"120*x1*y1*z1 + 60*I*x1*y1*z2 + 360*I*x1*y2*z1 - 180*x1*y2*z2 - 360*I*x2*y1*z1 + 180*x2*y1*z2 + 1080*x2*y2*z1 + 540*I*x2*y2*z2\n"}︡{"stdout":"120*x1*y1*z1 + 60*I*x1*y1*z2 + 360*I*x1*y2*z1 - 180*x1*y2*z2 - 360*I*x2*y1*z1 + 180*x2*y1*z2 + 1080*x2*y2*z1 + 540*I*x2*y2*z2\n"}︡{"stdout":"w= 0.004859840420526099824060786 + 0.05841924398625429553264605*I\n"}︡{"done":true}
︠5c3dcba9-9c2e-444b-a786-c5d88373f43e︠
%md 
## SymPy’s integrals
- https://docs.sympy.org/latest/modules/integrals/integrals.html
- [risk-engineering.org](https://risk-engineering.org/notebook/monte-carlo-LHS.html)

Let’s start with a simple integration problem in 1D,

$$\int_0^1 e^x dx$$
 
This is easy to solve analytically, and we can use the SymPy library in case you’ve forgotten how to resolve simple integrals.
︡711c57a0-2b20-483d-94da-ee892ed2d935︡{"done":true,"md":"## SymPy’s integrals\n- https://docs.sympy.org/latest/modules/integrals/integrals.html\n- [risk-engineering.org](https://risk-engineering.org/notebook/monte-carlo-LHS.html)\n\nLet’s start with a simple integration problem in 1D,\n\n$$\\int_0^1 e^x dx$$\n \nThis is easy to solve analytically, and we can use the SymPy library in case you’ve forgotten how to resolve simple integrals."}
︠465061dc-ea59-4fbd-91c3-7bb6123fb808s︠
import sympy
# we’ll save results using different methods in this data structure, called a dictionary
result = {}  
x = sympy.Symbol("x")
i = sympy.integrate(exp(x))
print(i)
result["analytical"] = float(i.subs(x, 1) - i.subs(x, 0))
print("Analytical result: {}".format(result["analytical"]))
︡1e9d8c92-d757-48ef-9e4c-db79bb063a99︡{"stdout":"exp(x)\n"}︡{"stdout":"Analytical result: 1.7182818284590453\n"}︡{"done":true}
︠c74e8a7c-19f4-4ecd-9912-edddaf85a486r︠
%md 

**Integrating with Monte Carlo** 
We can estimate this integral using a standard Monte Carlo method, where we use the fact that the expectation of a random variable is related to its integral

$$\mathbb{E}(f(x)) = \int_I f(x) dx $$

We will sample a large number N of points in I and calculate their average, and multiply by the range over which we are integrating.
<img src="https://raw.githubusercontent.com/tvcastillod/AlgorithmsUN2020II/master/SymPyLab/sympylabwolfram8.png" />
︡7e2b7024-3cd1-44c0-9e9d-c130a8591fce︡{"md":"<img src=\"https://raw.githubusercontent.com/tvcastillod/AlgorithmsUN2020II/master/SymPyLab/sympylabwolfram8.png\" />"}︡{"done":true}
︠942246c0-f932-4349-85f4-cff9392f722bs︠
import numpy
N = 10_000
accum = 0
for i in range(N):
    x = numpy.random.uniform(0, 3/2)
    accum += exp(x)*sin(2*x)
volume = 3/2
result["MC"] = volume * accum / float(N)
print("Standard Monte Carlo result: {}".format(result["MC"]))
︡060de498-5641-4fe1-8a44-ec0b1cae7b4d︡
︠95fb39a2-ec34-4dde-9415-1166afbfd579s︠
import sympy
x = Symbol("x")
i = integrate(exp(x)*sin(2*x))
print(i)
print(float(i.subs(x, 3/2) - i.subs(x, 0)))
︡6de193d4-1609-4766-a49e-8717d2b1bbe2︡{"stdout":"exp(x)*sin(2*x)/5 - 2*exp(x)*cos(2*x)/5\n"}︡{"stdout":"2.301226620237949\n"}︡{"done":true}
︠4538d42c-5497-41e8-8366-1d93ff53df4ds︠
import numpy
N = 10_000
accum = 0
l =[]
for i in range(N):
    x = numpy.random.uniform(0, 3/2)
    accum += exp(x)*sin(2*x)
volume = 3/2
result["MC"] = volume * accum / float(N)
print("Standard Monte Carlo result: {}".format(result["MC"]))
︡5ee03973-b403-4db5-9947-4f13b7be04bd︡{"stdout":"Standard Monte Carlo result: 2.30042184840401\n"}︡{"done":true}
︠8d7e94d4-d1c7-4ece-8c07-99158523aaac︠
%md 

**A higher dimensional integral** 

[risk-engineering.org](https://risk-engineering.org/notebook/monte-carlo-LHS.html) 

Let us now analyze an integration problem in dimension 4, the Ishigami function. This is a well-known function in numerical optimization and stochastic analysis, because it is very highly non-linear.

%md <img src="https://raw.githubusercontent.com/tvcastillod/AlgorithmsUN2020II/master/SymPyLab/sympylabwolfram9.png" />
︡22dca2b9-17e1-42be-9da4-fa8be2de496b︡{"done":true,"md":"\n**A higher dimensional integral** \n\n[risk-engineering.org](https://risk-engineering.org/notebook/monte-carlo-LHS.html) \n\nLet us now analyze an integration problem in dimension 4, the Ishigami function. This is a well-known function in numerical optimization and stochastic analysis, because it is very highly non-linear.\n\n%md <img src=\"https://raw.githubusercontent.com/tvcastillod/AlgorithmsUN2020II/master/SymPyLab/sympylabwolfram9.png\" />"}
︠f0ca5f19-699d-4400-9743-f475dc193090s︠
import sympy

x1 = sympy.Symbol("x1")
x2 = sympy.Symbol("x2")
x3 = sympy.Symbol("x3")
expr = x1*x2*x3 + x3**2 - x1**5
res = sympy.integrate(expr,
                      (x1, 0, 1),
                      (x2, 0, 2),
                      (x3, 0, 3))
# Note: we use float(res) to convert res from symbolic form to floating point form
result = {} 
result["analytical"] = float(res)
print("Analytical result: {}".format(result["analytical"]))
︡29ae2d3f-b54f-4487-9f49-c32c4b8c033e︡{"stdout":"Analytical result: 21.5\n"}︡{"done":true}
︠183f09fb-4925-4d5c-8c1f-d93b0f2b096es︠
N = 100_000
accum = 0
for i in range(N):
    xx1 = numpy.random.uniform(0,1)
    xx2 = numpy.random.uniform(0,2)
    xx3 = numpy.random.uniform(0,3)
    accum += (xx1*xx2*xx3 + xx3**2 - xx1**5)
volume = 6 #(2 * numpy.pi)**3
result = {} 
result["MC"] = volume * accum / float(N)
print("Standard Monte Carlo result: {}".format(result["MC"]))
︡1d51a799-c3b8-4be5-8483-34d5083c40ce︡{"stdout":"Standard Monte Carlo result: 21.4521693184033\n"}︡{"done":true}
︠71db003a-2681-45b7-bb09-bf36bcc3be87s︠
import math
import numpy
# adapted from https://mail.scipy.org/pipermail/scipy-user/2013-June/034744.html
def halton(dim: int, nbpts: int):
    h = numpy.full(nbpts * dim, numpy.nan)
    p = numpy.full(nbpts, numpy.nan)
    P = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    lognbpts = math.log(nbpts + 1)
    for i in range(dim):
        b = P[i]
        n = int(math.ceil(lognbpts / math.log(b)))
        for t in range(n):
            p[t] = pow(b, -(t + 1))

        for j in range(nbpts):
            d = j + 1
            sum_ = math.fmod(d, b) * p[0]
            for t in range(1, n):
                d = math.floor(d / b)
                sum_ += math.fmod(d, b) * p[t]

            h[j*dim + i] = sum_
    return h.reshape(nbpts, dim)
︡d9c3a218-e337-43f3-a1bb-1e2e50925e10︡{"done":true}
︠ef4a9575-36a0-49f8-9267-8d75027db96es︠
import matplotlib.pyplot as plt
import random
N = 1000
seq = halton(2, N)
plt.title("2D Halton sequence")
# Note: we use "alpha=0.5" in the scatterplot so that the plotted points are semi-transparent
# (alpha-transparency of 0.5 out of 1), so that we can see when any points are superimposed.
plt.axes().set_aspect('equal')
plt.scatter(seq[:,0], seq[:,1], marker=".", alpha=0.5);
︡a1ecfe80-0bde-4435-bad1-bb24475bc349︡{"stdout":"Text(0.5, 1.0, '2D Halton sequence')\n"}︡{"stdout":"<matplotlib.collections.PathCollection object at 0x7fa53e4f5f10>\n"}︡{"done":true}
︠429aac85-bfa4-4931-8138-33ca9f1ed876s︠
N = 10_000

seq = halton(3, N)
accum = 0
for i in range(N):
    xx1 = seq[i][0] * 1
    xx2 = seq[i][1] * 2
    xx3 = seq[i][2] * 3
    accum += xx1*xx2*xx3 + xx3**2 - xx1**5
volume = 6 
result = {} 
result["MC"] = volume * accum / float(N)
print("Qausi Monte Carlo Halton Sequence result: {}".format(result["MC"]))
︡8896571c-16b3-49f8-9272-852d885f9395︡{"stdout":"Qausi Monte Carlo Halton Sequence result: 21.4885326378350\n"}︡{"done":true}
︠2b681be7-a95c-40da-99ae-9f7ba692fa9a︠
%md 
   ## Wolfram alpha answers question in natural languaje
 - what is the circumference of the earth
 
 <img src="https://raw.githubusercontent.com/tvcastillod/AlgorithmsUN2020II/master/SymPyLab/sympylabwolfram7.png" />
︡c9c43896-0f38-4c4e-9571-695ed97c92ef︡{"done":true,"md":"   ## Wolfram alpha answers question in natural languaje\n - what is the circumference of the earth\n \n <img src=\"https://raw.githubusercontent.com/tvcastillod/AlgorithmsUN2020II/master/SymPyLab/sympylabwolfram7.png\" />"}
︠7db7ef55-1a9b-4406-9b31-cb497be60ba4︠









