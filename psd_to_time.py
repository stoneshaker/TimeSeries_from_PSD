#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
plt.show()


# In[8]:


gfg_inversed = ifft(nums)


# In[9]:


plt.plot(abs(gfg_inversed))
plt.show()


# In[ ]:




