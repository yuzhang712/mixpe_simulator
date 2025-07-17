import matplotlib.pyplot as plt
import math
import numpy as np

from matplotlib.pyplot import MultipleLocator

# fp16*fp16
y_fp16 = [4.5*4.5]
x_fp16 = [math.log2(1)]

#int4*fp16 pe
y_int4fp16 = [4.5*1.5]
x_int4fp16 = [math.log2(1)]

# int4*fp16 mixpe
y_mixfp = [4.5*1.5]
x_mixfp = [math.log2(4.82)]

# int8*int8
y_int8fp = [3*3]
x_int8fp = [math.log2(58.28)]

# int4*int8 pe
y_int4int8 = [1.5*3]
x_int4int8 = [math.log2(58.28)]

# int4*int8 mixpe
y_mixint = [1.5*3]
x_mixint = [math.log2(133.38)]

x = x_fp16+x_int4fp16+x_int8fp+x_int4int8
y = y_fp16+y_int4fp16+y_int8fp+y_int4int8

x_mix = x_mixfp+x_mixint
y_mix = y_mixfp+y_mixint

s1 = plt.scatter(x, y, marker='+', color = 'grey')
s2 = plt.scatter(x_mix, y_mix, marker='o', color = 'red')

plt.ylabel("Quantization SNR (dB)", fontdict={'family' : 'Times New Roman', 'size'   : 16})
plt.xlabel("Normalized Area Power Efficiency Product", fontdict={'family' : 'Times New Roman', 'size'   : 16})

ax=plt.gca()
x_major_locator=MultipleLocator(4)
# x_ticks = np.arange(-1, 8, 4)
ax.xaxis.set_major_locator(x_major_locator)
plt.xlim(-1, 9)
plt.xticks([0,4,8], ['2^0', '2^4', '2^8'], fontproperties = 'Times New Roman', size = 14)

y_major_locator=MultipleLocator(10)
# x_ticks = np.arange(-1, 8, 4)
ax.yaxis.set_major_locator(y_major_locator)
plt.ylim(0, 25)
plt.yticks([0,10,20], fontproperties = 'Times New Roman', size = 14)
plt.savefig("./dse.png", bbox_inches = 'tight')