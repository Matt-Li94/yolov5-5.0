from codecs import BOM_UTF16
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from sympy import *

plt.switch_backend('TkAgg')

# 篮筐高度
hoop_height = 3.05
# 投篮距离（三分线）
hoop_distance = 6.75
# 你的身高
your_height = 1.75
# 你的小臂长度
your_forearm_length = 0.27
# 你的投篮起跳高度
jump_height = 0.10
# 出手高度近似等于 身高+小臂+起跳高度，误差不大
height = your_height + your_forearm_length + jump_height

print('your height: ', your_height)
print('your forarm length: ', your_forearm_length)
print('your jump height:', jump_height)
print('南丰河村勇辉专属专属basketball paremeter')

a = Symbol('a')
b = Symbol('b')

r = solve([2*a*hoop_distance + b + math.tan(47/180*math.pi), a*hoop_distance**2 + b*hoop_distance-(hoop_height-height)], [a, b])
print('(a,b) = ', r[a], r[b])
print('alpha = ', math.atan(r[b]) / math.pi * 180)

x = np.linspace(0, hoop_distance, 50)
y = r[a]*x*x + r[b]*x

plt.ylim(0, hoop_height+2)
plt.plot(x,y+height)



plt.show()
