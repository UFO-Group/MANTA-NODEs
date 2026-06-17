#!/usr/bin/env python
# coding: utf-8

# In[323]:



import numpy as np
import numpy.random as npr
#import tensorflow as tf
import math
#import tensorflow.contrib.eager as tfe

#tf.enable_eager_execution()


# In[324]:


#pop0=np.loadtxt('pop0.txt')
#pop1=np.loadtxt('pop1.txt')
#pop2=np.loadtxt('pop2.txt')


# In[325]:



#time=np.linspace(0,100,201)

####处理population，转化成平均布局数
def pop_average(pop_discrete):
    length_time=pop_discrete.shape[0]
    num_trj=pop_discrete.shape[1]
    pop_init=np.zeros(length_time)
    for i in range(num_trj):
        pop_init=pop_init+pop_discrete[:,i]
    pop_ave=pop_init/num_trj
    return pop_ave
    


# In[326]:


#pop_ave0=pop_average(pop0)
#pop_ave1=pop_average(pop1)
#pop_ave2=pop_average(pop2)
#plt.plot(time,pop_ave0)
#plt.plot(time,pop_ave1)
#plt.plot(time,pop_ave2)


# In[327]:


#E0=np.loadtxt('energy0.txt')
#E1=np.loadtxt('energy1.txt')
#E2=np.loadtxt('energy2.txt')

##制作diabatic的能量gap


def energygap12_transf(E_high,E_low,pop_high):  #pop_high=pop2
    num_trj=E_high.shape[1]
    length_time=E_high.shape[0]
    DE=np.array([])
    for N in range(num_trj):
        dE_trj=np.array([])
        for i in range(length_time):
            if pop_high[i,N] > 0.5:
                dE=E_high[i,N]-E_low[i,N]
            else:
                dE=E_low[i,N]-E_high[i,N]
            dE_trj=np.append(dE_trj,dE)
        DE=np.append(DE,dE_trj)
    DE=DE.reshape(num_trj,length_time)
    DE=np.transpose(DE)
    return DE


def energygap01_transf(E_high,E_low,pop_low):    #pop_low=pop0
    num_trj=E_high.shape[1]
    length_time=E_high.shape[0]
    DE=np.array([])
    for N in range(num_trj):
        dE_trj=np.array([])
        for i in range(length_time):
            if pop_low[i,N] < 0.5:
                dE=E_high[i,N]-E_low[i,N]
            else:
                dE=E_low[i,N]-E_high[i,N]
            dE_trj=np.append(dE_trj,dE)
        DE=np.append(DE,dE_trj)
    DE=DE.reshape(num_trj,length_time)
    DE=np.transpose(DE)
    return DE

def nac_veloc(nac,veloc,num_atom):
    length_time=int(nac.shape[0]/num_atom)
    #nac=tf.constant(nac,dtype=float)
    #veloc=tf.constant(veloc,dtype=float)
    dnac_dt_trj=np.array([])
    for i in range(length_time):
        dnac_dt=np.sum(nac[num_atom*i:num_atom*i+num_atom,:]*veloc[num_atom*i:num_atom*i+num_atom,:])
        dnac_dt_trj=np.append(dnac_dt_trj,dnac_dt)
    return dnac_dt_trj

#def dnac_dt_calc_rescale(nac,veloc):
#    dnac_dt_trj=np.array([])
#    for i in range(201):
        
#        dnac_dt=np.sum(nac[6*i:6*i+6,:]*veloc[6*i:6*i+6,:])/math.sqrt(np.sum(veloc[6*i:6*i+6,:]**2))
#        dnac_dt_trj=np.append(dnac_dt_trj,dnac_dt)
#    return dnac_dt_trj

#def nac_mode(nac):
#    nac_modes_trj=np.array([])
#    for i in range(201):
#        nac_modes=np.sum(nac[6*i:6*i+6,:]*nac[6*i:6*i+6,:])
#        nac_modes_trj=np.append(nac_modes_trj,nac_modes)
#    return nac_modes_trj


# In[328]:


#DE12=energygap12_transf(E2,E1,pop2)
#DE01=energygap01_transf(E1,E0,pop0)
#plt.plot(time,DE01)
#plt.plot(time,DE12)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[248]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




