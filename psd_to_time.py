#!/usr/bin/env python
# coding: utf-8

# In[1]:


from random import random
import scipy as sp
import scipy.interpolate
from scipy.fft import fft, ifft
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def log_interp(zz,xx,yy,):
    logz = np.log10(zz)
    logx = np.log10(xx)
    logy = np.log10(yy)
    return np.power(10.0, np.interp(logz, logx, logy))


# In[3]:


freq = [20,80,350,2000]
amp = [0.01,0.04,0.04,0.007042]


# In[4]:


points = range(freq[0],freq[-1])


# In[5]:


nums = []
count = 0
lowFreq = freq[count]
highFreq = freq[count + 1] # no error checks for now
lowAmp = amp[count]
highAmp = amp[count + 1]
for j in range(0,len(points)):
    if points[j] == highFreq:
        temp = highAmp
    else:
      if points[j] > highFreq:
        count = count + 1
        lowFreq = freq[count]
        highFreq = freq[count + 1] # no error checks for now
        lowAmp = amp[count]
        highAmp = amp[count + 1]
      temp = log_interp(points[j],[lowFreq,highFreq],[lowAmp,highAmp])
    nums.append(temp)


# In[6]:


plt.plot(points,nums)
plt.show()


# In[7]:


plt.loglog(points,nums)
plt.title('Interppolated PSD')
plt.show()


# In[8]:
yuck = np.zeros(10000)
duh = np.ones(10000)
for j in range(1,len(yuck)):
  yuck[j] = np.random.normal(0.0,1.0)
  duh[j] = duh[j] * np.sin(2*3.14159265*100*j/1000)
  yuck[j] = yuck[j] + duh[j]

halfRange = len(nums)
print(halfRange)
for j in range(1,halfRange):
  print(halfRange-j+1)
  temp = nums[halfRange-j]
  nums.append(temp)

dummy = fft(yuck)
plt.plot(yuck)
plt.title('Time Series')
plt.show()
#plt.semilogy(abs(dummy[0:len(dummy)//2]))
plt.semilogy(abs(dummy))
plt.title('FFT')
plt.show()

gfg_inversed = ifft(nums)
#gfg_inversed = ifft(dummy)

# In[9]:
plt.semilogy(nums)
plt.show()

plt.plot(abs(gfg_inversed))
plt.title('Time Series from PSD')
plt.show()

print('Length of IFFT result =',len(gfg_inversed))
print('Nums[0] = ',nums[0])
print('Nums[end] = ',nums[len(points)])
print('Nums[-1] = ',nums[-1])

print('Length of IFFT result =',len(dummy))
print('Nums[0] = ',dummy[0])
print('Nums[end] = ',dummy[len(dummy)//2])
print('Nums[-1] = ',dummy[-1])
# In[ ]:




