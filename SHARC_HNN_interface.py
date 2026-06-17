#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np
#from matplotlib import pyplot as plt
import os
os.system("unset LD_LIBRARY_PATH")

#from GKDG import GaussianKDE
from torch.autograd.functional import jacobian
#from sklearn.neighbors import KernelDensity as KD
from efficient_kan import KAN

import os
import sys
import argparse
import math


import numpy as np



# In[2]:


device=torch.device("cpu")


def readfile(filename):
    try:
        f = open(filename)
        out = f.readlines()
        f.close()
    except IOError:
        print('File %s does not exist!' % (filename))
        sys.exit(12)
    return out


# In[16]:


def writefile(filename, content):
    # content can be either a string or a list of strings
    try:
        f = open(filename, 'w')
        if isinstance(content, list):
            for line in content:
                f.write(line)
        elif isinstance(content, str):
            f.write(content)
        else:
            print('Content %s cannot be written to file!' % (content))
            sys.exit(13)
        f.close()
    except IOError:
        print('Could not write to file %s!' % (filename))
        sys.exit(14)
        
        
def eformat(f, prec, exp_digits):
    '''Formats a float f into scientific notation with prec number of decimals and exp_digits number of exponent digits.

    String looks like:
    [ -][0-9]\\.[0-9]*E[+-][0-9]*

    Arguments:
    1 float: Number to format
    2 integer: Number of decimals
    3 integer: Number of exponent digits

    Returns:
    1 string: formatted number'''

    s = "% .*e" % (prec, f)
    mantissa, exp = s.split('e')
    return "%sE%+0*d" % (mantissa, exp_digits + 1, int(exp))


# In[17]:


def itmult(states):

    for i in range(len(states)):
        if states[i] < 1:
            continue
        yield i + 1
    return

# ======================================================================= #


def itnmstates(states):

    for i in range(len(states)):
        if states[i] < 1:
            continue
        for k in range(i + 1):
            for j in range(states[i]):
                yield i + 1, j + 1, k - i / 2.
    return


# In[18]:
Natoms = 8

states = [2,0,2]
sum_muti = sum(states) ## mutistate number 

nstates = 0
for imult, i, ims in itnmstates(states):
    print(imult, i, ims)
    nstates += 1
    


# In[3]:
Natoms = 8
cut_step = 1001


##units
scale_t=41.34110546116003  #Time : 1 fs = 41.34110546116003 a.u.
scale_m=1823.1814419193488 #Mass : 1 ram = 1823.1814419193488 a.u.
scale_L=1.8897259885789233 #coord : 1 Astrom = 1.8897259885789233 bohr
scale_E = 1/2625.5   #Energy : 1 kJ/mol = 1/2625.5 hartree 
ev_to_h = 0.03674930813664888 ## ev to a. u.


# In[4]:

toplist = np.array([[0,1,2,3],
                   [0,1,2,7],
                   [2,1,0,4],
                   [2,1,0,5],
                   [3,2,1,6]])

N_dih = toplist.shape[0]


def cross_mutip(a,b):

    T1=torch.transpose(torch.tensor([[[0., 1. ,0.],[0., 0., 1.],[1., 0., 0.]]]),1,2).to(device)
    T2=torch.transpose(torch.tensor([[[0., 0., 1.],[1., 0., 0.],[0., 1., 0.]]]),1,2).to(device)
    
    return torch.matmul(a,T1)*torch.matmul(b,T2) - torch.matmul(a,T2)*torch.matmul(b,T1)


def Internal_to_XYZ(Internal_all):

    i0, i1, i2 = toplist[0,0], toplist[0,1], toplist[0,2]
    
    N_dih = toplist.shape[0]
    
    ##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    N_batch = Internal_all.shape[0]
    b10 = Internal_all[:,:,0:1]
    b12= Internal_all[:,:,1:2]
    a012 = Internal_all[:,:,2:3]
    
    Translate = Internal_all[:,:,-3:]  ## bohr
    
    bond_list = Internal_all[:,:,3:3+N_dih]
    angle_list = Internal_all[:,:,3+N_dih:3+2*N_dih]
    dih_list = Internal_all[:,:,3+2*N_dih:3+3*N_dih]
    ##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    Rotate_θ = Internal_all[:,:,-6:-5]  ### θ φ β (rad) 
    Rotate_φ = Internal_all[:,:,-5:-4]
    Rotate_β = Internal_all[:,:,-4:-3]
    
    R_z = torch.cos(Rotate_θ)*torch.tensor([[1.,0.,0.],
                                          [0.,1.,0.],
                                          [0.,0.,0.]]).to(device) + \
    torch.sin(Rotate_θ)*torch.tensor([[0.,1.,0.],
                                     [-1.,0.,0.],
                                     [0.,0.,0.]]).to(device) + \
    torch.tensor([[0.,0.,0.],
                 [0.,0.,0.,],
                 [0.,0.,1.]]).to(device)
    
    
    R_y = torch.cos(Rotate_φ)*torch.tensor([[1.,0.,0.],
                                          [0.,0.,0.],
                                          [0.,0.,1.]]).to(device) + \
    torch.sin(Rotate_φ)*torch.tensor([[0.,0.,1.],
                                     [0.,0.,0.],
                                     [-1.,0.,0.]]).to(device) + \
    torch.tensor([[0.,0.,0.],
                 [0.,1.,0.,],
                 [0.,0.,0.]]).to(device)
    
    R_x = torch.cos(Rotate_β)*torch.tensor([[0.,0.,0.],
                                          [0.,1.,0.],
                                          [0.,0.,1.]]).to(device) + \
    torch.sin(Rotate_β)*torch.tensor([[0.,0.,0.],
                                     [0.,0.,1.],
                                     [0.,-1.,0.]]).to(device) + \
    torch.tensor([[1.,0.,0.],
                 [0.,0.,0.,],
                 [0.,0.,0.]]).to(device)
    
    
    xyzall = torch.zeros([N_batch,Natoms,3]).to(device)
    
    A_j_ = torch.zeros([N_batch,1,3]).to(device)
    A_k_ = torch.tensor([[1.,0.,0.]]).to(device)*b12
    A_i_ = torch.tensor([[1.,0.,0.]]).to(device)*b10*torch.cos(a012) + \
    torch.tensor([[0.,1.,0.]]).to(device)*b10*torch.sin(a012)
    
    ##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    xyzall[:,i0:i0+1,:] = torch.matmul(torch.matmul(torch.matmul(A_i_,R_x),R_y),R_z)
    xyzall[:,i1:i1+1,:] = torch.matmul(torch.matmul(torch.matmul(A_j_,R_x),R_y),R_z)
    xyzall[:,i2:i2+1,:] = torch.matmul(torch.matmul(torch.matmul(A_k_,R_x),R_y),R_z)
    ##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    index=0
    
    for list_top in toplist:
    
        A_i = xyzall[:,list_top[0]:list_top[0]+1,:]
        A_j = xyzall[:,list_top[1]:list_top[1]+1,:]
        A_k = xyzall[:,list_top[2]:list_top[2]+1,:]
        
        ### calc base_vector (ex,ey,ez)
        e_x = (A_k - A_j)/torch.sqrt(torch.matmul(torch.square(A_k - A_j),torch.ones([1,3,1]).to(device)))
        e_ji = (A_i - A_j) #/torch.sqrt(torch.matmul(torch.square(A_i - A_j),torch.ones([1,3,1]).to(device)))
        
        n_z = cross_mutip(e_x,e_ji)
        e_z = n_z/torch.sqrt(torch.matmul(torch.square(n_z),torch.ones([1,3,1]).to(device)))
        e_y  = cross_mutip(e_z, e_x)
        #############################
        
        b_ = bond_list[:,:,index:index+1]          ## bond (k--l)           ## bohr
        a_ = angle_list[:,:,index:index+1]         ## angle(j--k--l)        ## rad
        d_ = dih_list[:,:,index:index+1]           ## dihedral (i--j--k--l) ## rad
    
        A_l = A_k + b_*torch.cos(math.pi-a_)*e_x + \
        b_*torch.sin(math.pi-a_)*torch.cos(d_)*e_y +\
        b_*torch.sin(math.pi-a_)*torch.sin(d_)*e_z
        
        xyzall[:,list_top[3]:list_top[3]+1,:] = A_l
        
        index += 1
    
    xyz_bohr = xyzall + Translate  # coordinates
    
    return xyz_bohr #, e_x



def Batch_dXdQ(Q):
    N_sample=Q.shape[0]
    
    xyz_bohr = Internal_to_XYZ(Q)
    
    dXdQ_batch = torch.zeros([N_sample,Natoms,3,Natoms*3]).to(device)
    
    for I in range(Natoms): 
        for J in range(3):

            #torch.cuda.empty_cache()
            ID = 3*I+J
            
            
            gradients_temp = torch.nn.functional.one_hot(torch.tensor([ID]), Natoms*3).reshape(Natoms,3).type(torch.float32).to(device)
            gradients_temp = gradients_temp.unsqueeze(0)
            gradients_temp = gradients_temp.repeat(N_sample,1,1)
            
            #xyz_bohr.backward(gradients_temp,create_graph=True)
            
            dXdQ_compoents = torch.autograd.grad(xyz_bohr, Q, grad_outputs=gradients_temp,create_graph=True)[0]
            
            dXdQ_batch[:, I:I+1, J:J+1,:] = dXdQ_compoents.unsqueeze(2)
            
            #torch.cuda.empty_cache()
            
            #Q.grad.zero_()
            
    return dXdQ_batch, xyz_bohr





def cross_mutip_(a,b):
    T1=torch.transpose(torch.tensor([[0., 1. ,0.],[0., 0., 1.],[1., 0., 0.]]),0,1).to(device)
    T2=torch.transpose(torch.tensor([[0., 0., 1.],[1., 0., 0.],[0., 1., 0.]]),0,1).to(device)
    
    return torch.matmul(a,T1)*torch.matmul(b,T2) - torch.matmul(a,T2)*torch.matmul(b,T1)

def calc_angle(A1,A2,A3):
    V1=(A2-A1)/torch.sqrt(torch.matmul((A2-A1)**2,torch.ones([3,1]).to(device) ))
    V2=(A2-A3)/torch.sqrt(torch.matmul((A2-A3)**2,torch.ones([3,1]).to(device) ))
    return torch.acos(torch.matmul(V1*V2,torch.ones([3,1]).to(device) ))

def calc_length_bond(A1,A2):
    return torch.sqrt(torch.matmul((A1-A2)**2,torch.ones([3,1]).to(device) ))

def calc_cos_dihedral(A1,A2,A3,A4):

    n1=cross_mutip_((A2-A1)/torch.sqrt(torch.matmul((A2-A1)**2,torch.ones([3,1]).to(device))),(A3-A2)/torch.sqrt(torch.matmul((A3-A2)**2,torch.ones([3,1]).to(device) )))
    n1_normal=n1/torch.sqrt(torch.matmul(n1**2,torch.ones([3,1]).to(device)))
    n2=cross_mutip_((A3-A2)/torch.sqrt(torch.matmul((A3-A2)**2,torch.ones([3,1]).to(device))),(A4-A3)/torch.sqrt(torch.matmul((A4-A3)**2,torch.ones([3,1]).to(device) )))  #3 2; 4 3
    n2_normal=n2/torch.sqrt(torch.matmul(n2**2,torch.ones([3,1]).to(device) ))
    
    return torch.matmul(n1_normal*n2_normal,torch.ones([3,1]).to(device))

def calc_sin_dihedral(A1,A2,A3,A4):
    center_bond_vector=(A3-A2)/torch.sqrt(torch.matmul((A3-A2)**2,torch.ones([3,1]).to(device)))
    n1=cross_mutip_((A2-A1)/torch.sqrt(torch.matmul((A2-A1)**2,torch.ones([3,1]).to(device))),(A3-A2)/torch.sqrt(torch.matmul((A3-A2)**2,torch.ones([3,1]).to(device) )))
    n1_normal=n1/torch.sqrt(torch.matmul(n1**2,torch.ones([3,1]).to(device) ))
    n2=cross_mutip_((A3-A2)/torch.sqrt(torch.matmul((A3-A2)**2,torch.ones([3,1]).to(device))),(A4-A3)/torch.sqrt(torch.matmul((A4-A3)**2,torch.ones([3,1]).to(device) )))
    n2_normal=n2/torch.sqrt(torch.matmul(n2**2,torch.ones([3,1]).to(device) ))
    vec_n1_n2=cross_mutip_(n1_normal,n2_normal)
    
    return torch.matmul(vec_n1_n2*center_bond_vector,torch.ones([3,1]).to(device))


def calc_θ(cos_θ, sin_θ):
    
    if cos_θ > np.sqrt(2)/2:
        θ = float(np.arcsin(sin_θ)*180/math.pi)
    if sin_θ > np.sqrt(2)/2:
        θ = float(np.arccos(cos_θ)*180/math.pi)
    if cos_θ < -np.sqrt(2)/2:
        θ = float(180. - np.arcsin(sin_θ)*180/math.pi)
    if sin_θ < -np.sqrt(2)/2:
        θ = float(-np.arccos(cos_θ)*180./math.pi)
        
    return θ



def calc_Q_list(xyz):
    
    i0, i1, i2 = toplist[0,0], toplist[0,1], toplist[0,2]
    
    xyz_iconfig = xyz ## xyz is tensor (unit: bohr)
    
    Afour = xyz_iconfig[toplist[0],:]
    bond01 = calc_length_bond(Afour[0], Afour[1])   
    bond12 = calc_length_bond(Afour[1], Afour[2])
    angle012 = calc_angle(Afour[0], Afour[1], Afour[2]) #*180/np.pi
    ##@@@@@@@@@@@@@@@@@@@@@@@@
    
    bond01 = float(bond01.cpu().detach().numpy())
    bond12 = float(bond12.cpu().detach().numpy())
    angle012 = float(angle012.cpu().detach().numpy())
    
    Q_list = np.array([bond01, bond12, angle012])
    #Q_list
    #Q_list
    
    ## bond
    for index in range(toplist.shape[0]):
        #dih_traj = np.array([])
        xyz_iconfig = xyz
        Afour = xyz_iconfig[toplist[index],:]
        dih = calc_length_bond(Afour[2],Afour[3]).numpy()[0]
        
        Q_list = np.append(Q_list, dih)
        
    ## angle
    for index in range(toplist.shape[0]):
        #dih_traj = np.array([])
        xyz_iconfig = xyz
        Afour = xyz_iconfig[toplist[index],:]
        dih = calc_angle(Afour[1],Afour[2],Afour[3]).numpy()[0]  #*180/math.pi
        
        Q_list = np.append(Q_list, dih)
        
    ## dihedral
    
    for index in range(toplist.shape[0]):
        #dih_traj = np.array([])
        xyz_iconfig = xyz
        Afour = xyz_iconfig[toplist[index],:]
        sin_ = calc_sin_dihedral(Afour[0],Afour[1],Afour[2],Afour[3])
        cos_ = calc_cos_dihedral(Afour[0],Afour[1],Afour[2],Afour[3])
        dih = calc_θ(cos_.numpy(), sin_.numpy())*math.pi/180.
        
        Q_list = np.append(Q_list, dih)
    
        
    ## RT matrix
    
    coord_i = xyz.cpu().detach().numpy()# coord_itraj[nstep,:,:]
    
    ####@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    ####@@@@@@@@@@@@@@ [i0,i1,i2] = [0,1,2] in original setting @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    ####@@@@@@@@@@@@@@ [i0,i1,i2] = toplist[0,0], toplist[0,1], toplist[0,2] now #####################################################
    T_XYZ = coord_i[i1,:]
    vec_12 = (coord_i[i2:i2+1,:]-coord_i[i1:i1+1,:])/np.sqrt(np.sum((coord_i[i2:i2+1,:]-coord_i[i1:i1+1,:])**2))
    ###@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    cos_θ, sin_θ = vec_12[:,0]/np.sqrt(vec_12[:,0]**2 + vec_12[:,1]**2), vec_12[:,1]/np.sqrt(vec_12[:,0]**2 + vec_12[:,1]**2)
    cos_φ, sin_φ = vec_12[:,2], np.sqrt(vec_12[:,0]**2 + vec_12[:,1]**2)
    
    
    θ = calc_θ(cos_θ, sin_θ)
    θneg = (-θ)*math.pi/180.
    
    φ = calc_θ(cos_φ, sin_φ)
    φneg = (φ-90.)*math.pi/180.
    
    R_z = torch.tensor([[math.cos(θneg), -math.sin(θneg), 0.],
                        [math.sin(θneg), math.cos(θneg), 0.],
                        [0., 0., 1.]])
    
    R_y = torch.tensor([[math.cos(φneg),0, -math.sin(φneg)],
                        [0., 1., 0.],
                        [math.sin(φneg),0, math.cos(φneg)]])
    
    ###@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    X_primitive = torch.transpose(torch.tensor(np.float32(coord_i - coord_i[i1:i1+1,:])),0,1)
    ###@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    X_middle = torch.matmul(R_y,torch.matmul(R_z,X_primitive))
    coord_i_new = np.transpose(X_middle.numpy())
    
    ###@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    vec_10 = (coord_i_new[i0:i0+1,:]-coord_i_new[i1:i1+1,:])/np.sqrt(np.sum((coord_i_new[i0:i0+1,:]-coord_i_new[i1:i1+1,:])**2))
    ###@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    cos_β, sin_β = vec_10[:,1]/np.sqrt(vec_10[:,1]**2 + vec_10[:,2]**2), vec_10[:,2]/np.sqrt(vec_10[:,1]**2 + vec_10[:,2]**2)
    
    β = calc_θ(cos_β, sin_β)
    βneg = (-β)*math.pi/180.
    
    ### for later calculation
    θneg, φneg, βneg = -θneg, -φneg, -βneg
    
    Q_list = np.append(Q_list,θneg)
    Q_list = np.append(Q_list, φneg)
    Q_list = np.append(Q_list, βneg)
    Q_list = np.append(Q_list,T_XYZ)
    
    return Q_list
#
  

class EG(nn.Module):
    def __init__(self):
        super(EG, self).__init__()
        
        self.layerkan = KAN([3+4*N_dih, 2*(3+4*N_dih)+1, 3+4*N_dih, 1], base_activation=nn.Identity) #KAN(width=[3+4*N_dih,3+4*N_dih,1,1], grid=5, k=3, seed=0)
        #self.layerkan = KAN([3+4*N_dih, 3+4*N_dih, 1, 1], base_activation=nn.Identity)
    
    def forward(self, inputs_): # inputs_ is Q [Ns x 3*Natoms]
        
        x = inputs_ 
        
        x_I = x[:,:3]          
        x_b = x[:,3:3+N_dih]
        x_a = torch.cos(x[:,3+N_dih:3+2*N_dih])
        x_d = torch.concat((torch.cos(x[:,3+2*N_dih:3+3*N_dih]),torch.sin(x[:,3+2*N_dih:3+3*N_dih])),axis=1)
        
        x_ = torch.concat((x_I,x_b,x_a,x_d),axis=1)
    
        z = self.layerkan(x_)
        
        return z
    

class Hamiltonian(nn.Module):
    def __init__(self):
        super(Hamiltonian, self).__init__()
        
        self.CombNets = nn.ModuleList(EG() for i_ in range( sum_muti ))
        
    def forward(self, x):  ## x is Q [Ns x 3*Natoms]
        
        Nsamples = x.shape[0]
        
        y = torch.zeros([Nsamples, sum_muti]).to(device)
        dy = torch.zeros([Nsamples, sum_muti, Natoms*3]).to(device)
        
        for istate in range(sum_muti):
            y_s = self.CombNets[istate](x)
            dy_s = torch.autograd.grad(y_s, x, grad_outputs=torch.ones(y_s.size()).to(device), create_graph=True)[0]
            
            y[:, istate:istate+1] = y_s                   # torch.concat((y, y_s),axis=0)
            dy[:, istate] = dy_s                          #torch.concat((dy, dy_s), axis=0)
            
        return y, dy                                      #_s.unsqueeze(1)#, dy[:, istate:istate+1, Natoms, 3]
    


class Olp_Net(nn.Module):
    def __init__(self):
        super(Olp_Net, self).__init__()

        n_neurons = 500#
        
        #
        self.layer_d1=nn.Linear( 2*(N_dih*4+3),n_neurons)
        self.layer_d1x=nn.Linear(N_dih*4+3,n_neurons)
        self.layer_d1y=nn.Linear(N_dih*4+3,n_neurons)
        
        self.layer_d2=nn.Linear(n_neurons,n_neurons)
        self.layer_d3=nn.Linear(n_neurons,n_neurons)
        self.layer_d4=nn.Linear(n_neurons,n_neurons)
        self.layer_d5=nn.Linear(n_neurons,1)
        
        self.layer_g1=nn.Linear(1,n_neurons)  ## input gap
        self.layer_g2=nn.Linear(n_neurons,n_neurons)
        self.layer_g3=nn.Linear(n_neurons,n_neurons)
        self.layer_g4=nn.Linear(n_neurons, 1)
        
    
    def forward(self, inputs_, deltaE): ## inputs_ is the concat tensor of Q_I (last step) and Q_J (now step) //  conert to (x, y)
        
        x, y = inputs_[:,:Natoms*3], inputs_[:, Natoms*3:]
        
        activation = torch.nn.Tanh()# softplus_shift #spk.nn.activations.shifted_softplus  #
        #activation_ = spk.nn.activations.shifted_softplus
        
        x_I = x[:,:3]          
        x_b = x[:,3:3+N_dih]
        x_a = torch.cos(x[:,3+N_dih:3+2*N_dih])
        x_d = torch.concat((torch.cos(x[:,3+2*N_dih:3+3*N_dih]),torch.sin(x[:,3+2*N_dih:3+3*N_dih])),axis=1)
        
        y_I = y[:,:3]          
        y_b = y[:,3:3+N_dih]
        y_a = torch.cos(y[:,3+N_dih:3+2*N_dih])
        y_d = torch.concat((torch.cos(y[:,3+2*N_dih:3+3*N_dih]),torch.sin(y[:,3+2*N_dih:3+3*N_dih])),axis=1)
        
        x_ = torch.concat((x_I,x_b,x_a,x_d),axis=1)
        y_ = torch.concat((y_I,y_b,y_a,y_d),axis=1)
        
        g = activation(self.layer_g1(deltaE**2))
        g = activation(self.layer_g2(g))
        g = activation(self.layer_g3(g))
        g = activation(self.layer_g4(g))
        
        z = activation(self.layer_d1(torch.concat((x_,y_),axis=1)))
        z = activation(self.layer_d2(z))
        z = activation(self.layer_d3(z))
        z = activation(self.layer_d4(z))
        z = activation(self.layer_d5(z)/g)
        
        
        return z
    

class Overlap_Matrix(nn.Module):
    def __init__(self):
        super(Overlap_Matrix, self).__init__()
        
        self.ovnets = nn.ModuleList(Olp_Net() for i_ in range(2))
    
    def forward(self, x, gaps): ## inputs_ is the concat tensor of Q_I (last step) and Q_J (now step) //  conert to (x, y)
        ## gaps is [1x2]
        y_ = torch.tensor([]).to(device)
        
        for i in range(2):
            y = self.ovnets[i](x, gaps[:,i:i+1])
            y_ = torch.concat( (y_, y), axis=-1)
        
        return y_



def Pred_Hdiag(E_list,socs_list):

    Hsocs = torch.zeros([socs_list.shape[0], nstates, nstates]).type(torch.complex64).to(device)
    
    ## socs (T--T)
    Hsocs[:, states[0]::2, states[0]+1::2] = torch.tile(socs_list[:,-1,:].unsqueeze(-2), [1,3,1])
    
    ## socs (S--T)
    k=0
    for i_s in range(states[0]):
        for i_t in range(states[2]):
            Hsocs[:,i_s, i_t+states[0]::2] = socs_list[:,k,:]
            
            k+=1
        
    ## transpose and conj
    Hsocs_conj = torch.conj(torch.transpose(Hsocs,-1,-2))
    
    ## diagonal element of MCH rep Hamiltonian matrix
    ## E_list is from the Datasets (Data_E) [Ns x sum_muti]
    ## socs_list with shape [Ns x 5 x 3]
    selec_elist = np.int64(np.append(np.arange(sum_muti)[:states[0]],np.tile(np.arange(sum_muti)[states[0]:],3)))
    HmchE = E_list.unsqueeze(-1)[:, selec_elist]*torch.eye(nstates, nstates).unsqueeze(0).to(device)
    
    #En_pred = torch.linalg.eigvalsh( (Hsocs + Hsocs_conj + HmchE).cpu() ).to(device)
    
    return Hsocs + Hsocs_conj + HmchE



class NetQinp(nn.Module):
    def __init__(self, hidden_dims, num_layers):
        super(NetQinp, self).__init__()
        
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.active_funct = torch.nn.Tanh()#spk.nn.activations.shifted_softplus
        
        ## define layers
        input_dims = 3+4*N_dih
        output_dims = 3
        ########################
        
        self.input_layers = nn.Linear(input_dims, hidden_dims)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dims, hidden_dims) for _ in range(num_layers)])
        self.output_layers = nn.Linear(hidden_dims, output_dims)
        
    
    def forward(self,x):   ## input x is q
        
        x_I = x[:,:3]          
        x_b = x[:,3:3+N_dih]
        x_a = torch.cos(x[:,3+N_dih:3+2*N_dih])
        x_d = torch.concat((torch.cos(x[:,3+2*N_dih:3+3*N_dih]),torch.sin(x[:,3+2*N_dih:3+3*N_dih])),axis=1)
        
        x = torch.concat((x_I,x_b,x_a,x_d),axis=1)

        ###################################################
        x = self.active_funct(self.input_layers(x))
        
        for layers in self.hidden_layers:
            x = self.active_funct(layers(x))
        
        y = self.active_funct(self.output_layers(x))
        return y


class SOCsNet(nn.Module):
    def __init__(self):
        super(SOCsNet, self).__init__()
        
        self.NetR = nn.ModuleList(NetQinp(hidden_dims=200, num_layers=3) for i_ in range(5)) ## real _ part
        self.NetI = nn.ModuleList(NetQinp(hidden_dims=200, num_layers=3) for i_ in range(5)) ## image _ part
    
    def forward(self, x):  ## input x is q
        
        Nsamples = x.shape[0]
        y = torch.zeros([Nsamples, 5, 3]).type(torch.complex64).to(device)
        
        for i in range(5):
            yr = self.NetR[i](x)  ### [Ns x 3]            
            yi = self.NetI[i](x)  ### [Ns x 3]   
            y[:,i]  = yr + yi*1j
        
        return y

    
##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
def readfile(filename):
    try:
        f = open(filename)
        out = f.readlines()
        f.close()
    except IOError:
        print('File %s does not exist!' % (filename))
        sys.exit(12)
    return out


def eformat(f, prec, exp_digits):
    '''Formats a float f into scientific notation with prec number of decimals and exp_digits number of exponent digits.

    String looks like:
    [ -][0-9]\\.[0-9]*E[+-][0-9]*

    Arguments:
    1 float: Number to format
    2 integer: Number of decimals
    3 integer: Number of exponent digits

    Returns:
    1 string: formatted number'''

    s = "% .*e" % (prec, f)
    mantissa, exp = s.split('e')
    return "%sE%+0*d" % (mantissa, exp_digits + 1, int(exp))

def itmult(states):

    for i in range(len(states)):
        if states[i] < 1:
            continue
        yield i + 1
    return


def itnmstates(states):

    for i in range(len(states)):
        if states[i] < 1:
            continue
        for k in range(i + 1):
            for j in range(states[i]):
                yield i + 1, j + 1, k - i / 2.
    return


# In[75]:


QMin = readfile('./QM.in')

for iline, line in enumerate(QMin):
    if "unit angstrom" in line:
        s_factor = scale_L
    if "unit bohr" in line:
        s_factor = 1.
    if "step" in line:
        if QMin[iline].split()[-1] == 'samestep':
            istep = int(np.loadtxt('./Istep_index.txt'))
        else:
            istep = int(QMin[iline].split()[-1])


np.savetxt('./Istep_index.txt', np.array([istep]), delimiter='  ', fmt="%s")

xyz = np.zeros([Natoms,3])
    
for iatom in range(Natoms):
    iline = iatom+2
    for ixyz in range(3):
        icolumn = ixyz+1
        xyz[iatom,ixyz] = float(QMin[iline].split()[icolumn])*s_factor
        

xyz_now = torch.tensor(xyz).unsqueeze(0).type(torch.float32).to(device)  ## bohr
#xyz_now.requires_grad = True


outputdata = readfile('../output.dat')

if istep==0:
    xyz_last = xyz_now

else:

    for iline, line in enumerate(outputdata):
        if "Step" in line:
            if int(outputdata[iline+1])==istep-1:
            #ilocate = iline
            #print(iline)
                for Iter in range(iline,iline+116):
                    if "Geometry in" in outputdata[Iter]:  ## unit = a.u.
                        ilocate_geo = Iter
                        break
                        
                break
            


    xyz = np.zeros([Natoms,3])
    for iatom in range(Natoms):
        iline = iatom+ilocate_geo+1
        for ixyz in range(3):
            icolumn = ixyz
            xyz[iatom,ixyz] = float(outputdata[iline].split()[icolumn])
            
    xyz_last = torch.tensor(xyz).unsqueeze(0).type(torch.float32).to(device)


FitSOCs = torch.load('/home/dell/xuhy/Hamilton_Flow/acrole/KAN_model/CKPT_kcan/soc_nn', map_location=torch.device('cpu'))
VQ_S0 = Hamiltonian().to(device)
VQ_S0.load_state_dict( torch.load('/home/dell/xuhy/Hamilton_Flow/acrole/KAN_model/CKPT_kcan/HamKan_ckp_params_CKP_effcieKan_bigbatch_new') )
#torch.load('/home/dell/xuhy/Hamilton_Flow/acrole/KAN_model/CKPT_kcan/HamKan_ckp_params_CKP_effcieKan_bigbatch_new', map_location=torch.device('cpu'))
OMatrix = torch.load( '/home/dell/xuhy/Hamilton_Flow/acrole/KAN_model/CKPT_kcan/OMatrix_ckpt_ckp_', map_location=torch.device('cpu') )

Qb = calc_Q_list(xyz_last.detach()[0])
Qe = calc_Q_list(xyz_now.detach()[0])
Qb = torch.tensor(Qb.reshape(1,Natoms*3)).type(torch.float32).to(device)
Qe = torch.tensor(Qe.reshape(1,Natoms*3)).type(torch.float32).to(device)
Qb.requires_grad = True
Qe.requires_grad = True

E_list, dEdQ = VQ_S0(Qe)  ## E_list shape :[1 x 4]
dXdQ_, xyz_bohr =  Batch_dXdQ(Qe.unsqueeze(0))
dEdX = torch.matmul(dEdQ, torch.linalg.inv(dXdQ_.reshape(1, Natoms*3, Natoms*3))).reshape(1,4,Natoms,3) ## 1 x 4 x 3*Na

#torch.matmul(z_pred_[:,0:1,:], torch.linalg.inv(dXdQ_.reshape(cut_step, Natoms*3, Natoms*3)))

## socs
scale_socs = 1000
socs_matrix_list = FitSOCs(Qe)/scale_socs
HMCH_matrix = Pred_Hdiag(E_list, socs_matrix_list)[0].detach().numpy() ## complex numpy

## overlaps  
gap_list = torch.matmul(E_list, torch.tensor([[-1.,0.],[1.,0.],[0.,-1.],[0.,1.]]))
olp_origin = OMatrix( torch.concat( (Qb, Qe),axis=1), gap_list) ## 1 x 2
olp_origin = olp_origin[0]

#aaaa = torch.tensor([-1.3,12])
#aaaaaaaa
#signa_ = torch.sign(olp_origin[torch.where(torch.abs(olp_origin)>1)[0].detach().numpy()])

#olp_origin[torch.where(torch.abs(olp_origin)>1)[0].detach().numpy()] = signa_*1.  ## shape [1 x 2]




# In[ ]:





# In[ ]:





# In[125]:





# In[140]:


### write Hamiltonian matrices

#H_mat = E0.cpu().detach().numpy()*np.array([[1,0],[0,0]]) + E1.cpu().detach().numpy()*np.array([[0,0],[0,1]])

string=''
string += '! %i Hamiltonian Matrix (%ix%i, complex)\n' % (1, nstates, nstates)
string += '%i %i\n' % (nstates, nstates)
for i in range(nstates):
    for j in range(nstates):
        string += '%s %s ' % (eformat(HMCH_matrix.real[i,j], 12, 3), eformat(HMCH_matrix.imag[i,j], 12, 3))
    string += '\n'
string += '\n'


# In[141]:


### write Dipole

string += '! %i Dipole Moment Matrices (3x%ix%i, complex)\n' % (2, nstates, nstates)
for xyz in range(3):
    string += '%i %i\n' % (nstates, nstates)
    for i in range(nstates):
        for j in range(nstates):
            string += '%s %s ' % (eformat(0.0, 12, 3), eformat(0.0, 12, 3))
        string += '\n'
    string += ''
string += '\n'


# In[142]:


## write gradients
#states = [2,0,2]

string += '! %i Gradient Vectors (%ix%ix3, real)\n' % (3, nstates, Natoms)
i = 0
for imult, istate, ims in itnmstates(states):
    if imult==1:
        if istate==1: ## S0
            gradients_ = dEdX[0,0].detach().numpy()
        if istate==2: ## S1
            gradients_ = dEdX[0,1].detach().numpy()
            
    if imult==3:
        if istate==1: ## T1
            gradients_ = dEdX[0,2].detach().numpy()
        if istate==2: ## T2
            gradients_ = dEdX[0,3].detach().numpy()

    
    string += '%i %i ! %i %i %i\n' % (Natoms, 3, imult, istate, ims)

    for atom in range(Natoms):
        for xyz in range(3):

            string += '%s ' % (eformat(float(gradients_[atom,xyz]), 12, 3))
             
        string += '\n'
    string += ''
    i += 1

string += '\n'



# In[143]:


# write Overlap

OVMatrx = np.zeros([nstates,nstates])
OVMatrx[0,0] = torch.sqrt(torch.abs(1 - olp_origin[0]**2)).detach().numpy()
OVMatrx[1,1] = torch.sqrt(torch.abs(1 - olp_origin[0]**2)).detach().numpy()
OVMatrx[0,1] = olp_origin[0].detach().numpy()
OVMatrx[1,0] = -olp_origin[0].detach().numpy()

OVMatrx[2,2] = torch.sqrt(torch.abs(1 - olp_origin[1]**2)).detach().numpy()
OVMatrx[3,3] = torch.sqrt(torch.abs(1 - olp_origin[1]**2)).detach().numpy()
OVMatrx[2,3] = olp_origin[1].detach().numpy()
OVMatrx[3,2] = -olp_origin[1].detach().numpy()


OVMatrx[4,4] = torch.sqrt(torch.abs(1 - olp_origin[1]**2)).detach().numpy()
OVMatrx[5,5] = torch.sqrt(torch.abs(1 - olp_origin[1]**2)).detach().numpy()
OVMatrx[4,5] = olp_origin[1].detach().numpy()
OVMatrx[5,4] = -olp_origin[1].detach().numpy()


OVMatrx[6,6] = torch.sqrt(torch.abs(1 - olp_origin[1]**2)).detach().numpy()
OVMatrx[7,7] = torch.sqrt(torch.abs(1 - olp_origin[1]**2)).detach().numpy()
OVMatrx[6,7] = olp_origin[1].detach().numpy()
OVMatrx[7,6] = -olp_origin[1].detach().numpy()

if istep==0:
    string += '\n'

else:
    overlap_mat = OVMatrx  #np.zeros([nstates,nstates])
    #olp_ = OVLP( torch.concat((Q_last[:,0,:],Q[:,0,:]),axis=1) ).detach().numpy()
    
    #overlap_mat[0,0], overlap_mat[0,1] = olp_[:,0], olp_[:,1]
    #overlap_mat[1,0], overlap_mat[1,1] = -olp_[:,1], olp_[:,0]

    string += '! 6 Overlap matrix (%ix%i, complex)\n' % (nstates, nstates)
    string += '%i %i\n' % (nstates, nstates)
    for i in range(nstates):
        for j in range(nstates):
            string += '%s %s ' % (eformat(overlap_mat[i,j], 9, 3), eformat(0.0, 9, 3))
    
        string += '\n'

    string += '\n'


# In[144]:





# In[145]:


## phase

string += '! 7 Phases\n%i ! for all nmstates\n' % (nstates)
for i in range(nstates):
    string += '%s %s\n' % (eformat(1.0, 9, 3), eformat(0.0, 9, 3))
string += '\n'


# In[146]:


string += '! 8 Runtime'
string += '\n'
string += '%s ' % (eformat(0.0, 12, 3))
string += '\n'



with open('QM.out', 'w') as f:

    f.write(string)


# In[99]:


#gradient_S0


# In[147]:


#readfile('qm.out')


# In[ ]:





# In[ ]:





# In[19]:


#sin_.numpy()


# In[ ]:





# In[20]:





# In[21]:





# In[ ]:





# In[13]:


#Q_init = torch.tensor(np.float32(np.loadtxt('./Initcond_Q.txt').reshape(N_traj,1,Natoms*3))).to(device)
#Q_init.requires_grad = True

#P_init = torch.tensor(np.float32(np.loadtxt('./Initcond_P.txt').reshape(N_traj,1,Natoms*3))).to(device)
#P_init.requires_grad = True


# In[79]:


#Batch_dXdQ(Q_init)
#dXdQ_batch, xyz_bohr = Batch_dXdQ(Q_init)
#G = torch.matmul(torch.transpose(dXdQ_batch.reshape(N_traj,Natoms*3,Natoms*3),1,2), Mass*dXdQ_batch.reshape(N_traj,Natoms*3,Natoms*3))


# In[80]:


#G_inv = torch.linalg.inv(G)
#K = 0.5*torch.matmul(torch.matmul(P_init,G_inv),torch.transpose(P_init,1,2))

#dKdQ = torch.autograd.grad(K,Q_init,grad_outputs=torch.ones([N_traj,1,1]).to(device),create_graph=True)[0]


# In[81]:


#V1(Q_init[:,0,:]).shape
#V1(Q_init)

#Q_init.shape


# In[82]:


#dKdQ[0]

#torch.autograd.grad(V1(Q_init[:,0,:]), Q_init, grad_outputs=torch.ones([N_traj,1]).to(device))[0]


# In[ ]:





# In[46]:





# In[47]:





# In[ ]:





# In[36]:





# In[37]:


#dE0dQ


# In[38]:





# In[45]:


#gradient_S0


# In[148]:


#plt.scatter(gradient_S1, np.loadtxt('./train_sets_new/txt'))
#plt.plot(np.linspace(-0.15,0.20,3), np.linspace(-0.15,0.20,3))


# In[18]:


#def velert(q_b,p_b,h,istate):
    
    
#    dHdQ, dHdP, E, K, dXdQ_batch, xyz_bohr = Hamiltonian_Q(q_b, p_b, istate)
    #torch.cuda.empty_cache()
#    q_i = q_b + (h/2)*dHdP
    
#    dHdQ, dHdP, E, K, dXdQ_batch, xyz_bohr = Hamiltonian_Q(q_i, p_b, istate)
    #torch.cuda.empty_cache()
#    p_e = p_b - h*dHdQ
    
#    dHdQ, dHdP, E, K, dXdQ_batch, xyz_bohr = Hamiltonian_Q(q_i, p_e, istate)
    #torch.cuda.empty_cache()
#    q_e = q_i + (h/2)*dHdP
    
    #E_kin = 0.5*float(torch.matmul(torch.matmul(p_e[0],G_inv),torch.transpose(p_e[0],0,1)))
    
#    return q_e, p_e, E, K


# In[23]:


#N_traj = 1

#istate = np.concatenate((np.zeros([N_traj,1]),np.ones([N_traj,1])),axis=1)
#Q_init = torch.tensor(np.float32(np.loadtxt('./Initcond_Q.txt')[:N_traj].reshape(N_traj,1,Natoms*3))).to(device)
#Q_init.requires_grad = True

#P_init = torch.tensor(np.float32(np.loadtxt('./Initcond_P.txt')[:N_traj].reshape(N_traj,1,Natoms*3))).to(device)
#P_init.requires_grad = True


# In[22]:


#Q_init.shape


# In[17]:


#P_init.shape
#istate.shape
#np.loadtxt('./Initcond_Q.txt').shape


# In[24]:


#istate = np.concatenate((np.zeros([N_traj,1]),np.ones([N_traj,1])),axis=1)
#istate_traj = np.expand_dims(istate,0) #istate
#h=0.5*scale_t 
#Δt=h

#q_b, p_b = Q_init, P_init

#q_e, p_e, E_t, K = velert(q_b,p_b,h,istate)


# In[25]:


#q_e.shape, E_t.shape
#E_t


# In[74]:


#dHdQ, dHdP, E, K, dXdQ_batch, xyz_bohr = Hamiltonian_Q(Q_init, P_init, istate)


# In[73]:


#dHdQ.shape, dHdP.shape, E.shape, K.shape, dXdQ_batch.shape, xyz_bohr.shape


# In[100]:


#istate


# In[104]:


#istate = np.concatenate((np.zeros([N_traj,1]),np.ones([N_traj,1])),axis=1)
#istate_traj = np.expand_dims(istate,0) #istate#

#Δt=h

#hbar = 1.0
#Nsubstep = 21
#list_substep = np.linspace(0,1,Nsubstep)

#c0 = torch.zeros([N_traj,1]).type(torch.complex64).to(device)
#c1 = torch.sqrt(1 - c0*c0)

#q_traj = torch.transpose(Q_init,0,1)  #Q_init

#q_b,p_b = Q_init, P_init

#E_Δt = Energy(q_b)              ## -Δt
#E_2Δt = torch.ones_like(E_Δt)  ## -2Δt
#E_3Δt = torch.ones_like(E_Δt)  ## -3Δt
#σ_b = torch.ones_like(E_Δt) #TB_BA_coupling(E_3Δt,E_2Δt,E_Δt,E_t)

#E_b = E_Δt

#for istep in range(1,280):
    
    
#    print(istep)
#    q_e, p_e, E_t, K = velert(q_b,p_b,h,istate)
    #E_e = E_t
    
    #σ_e = TB_BA_coupling(E_3Δt,E_2Δt,E_Δt,E_t)
    
    #print(c0.shape)
    ##split hamiton
    #c0,c1,istate,p_e_correct = update_coefficient(c0,c1,E_b, E_e, σ_b, σ_e, list_substep, istate)
    
    #istate_traj = np.concatenate((istate_traj, np.expand_dims(istate,0)),axis=0)
#    q_traj = torch.concat((q_traj,torch.transpose(q_e,0,1)),axis=0)
    
    #torch.cuda.empty_cache()
    
#    q_b, p_b = q_e, p_e
    #E_b = E_e
   # σ_b = σ_e
   # E_3Δt,E_2Δt,E_Δt = E_2Δt,E_Δt,E_t


# In[102]:


#XYZ = Internal_to_XYZ(q_traj.reshape(20*31,1,54))


# In[93]:


#XYZ.shape
#BB = torch.sqrt(torch.sum((XYZ[:,10,:]-XYZ[:,14,:])**2,1))/scale_L


# In[103]:


#plt.plot(BB.reshape(20,31).cpu().detach().numpy(),c='grey')
#plt.xlim(-8,145)
#plt.ylim(1.3,1.7)


# In[35]:


## inter012


#xyz_iconfig = xyz ## xyz is tensor
#
#Afour = xyz_iconfig[toplist[0],:]
#bond01 = calc_length_bond(Afour[0], Afour[1])   
#bond12 = calc_length_bond(Afour[1], Afour[2])
#angle012 = calc_angle(Afour[0], Afour[1], Afour[2]) #*180/np.pi
###@@@@@@@@@@@@@@@@@@@@@@@@
#
#bond01 = float(bond01.cpu().detach().numpy())
#bond12 = float(bond12.cpu().detach().numpy())
#angle012 = float(angle012.cpu().detach().numpy())
#
#Q_list = np.array([bond01, bond12, angle012])
#
#
### bond
#for index in range(toplist.shape[0]):
#
#    xyz_iconfig = xyz
#    Afour = xyz_iconfig[toplist[index],:]
#    dih = calc_length_bond(Afour[2],Afour[3]).numpy()[0]
#    
#    Q_list = np.append(Q_list, dih)
#    
### angle
#for index in range(toplist.shape[0]):
#
#    xyz_iconfig = xyz
#    Afour = xyz_iconfig[toplist[index],:]
#    dih = calc_angle(Afour[1],Afour[2],Afour[3]).numpy()[0]  #*180/math.pi
#    
#    Q_list = np.append(Q_list, dih)
#    
### dihedral
#
#for index in range(toplist.shape[0]):
#
#    xyz_iconfig = xyz
#    Afour = xyz_iconfig[toplist[index],:]
#    sin_ = calc_sin_dihedral(Afour[0],Afour[1],Afour[2],Afour[3])
#    cos_ = calc_cos_dihedral(Afour[0],Afour[1],Afour[2],Afour[3])
#    dih = calc_θ(cos_.numpy(), sin_.numpy())*math.pi/180.
#    
#    Q_list = np.append(Q_list, dih)
#
#    
### RT matrix
#
#coord_i = xyz.cpu().detach().numpy()# coord_itraj[nstep,:,:]
#T_XYZ = coord_i[1,:]
#
#vec_12 = (coord_i[2:3,:]-coord_i[1:2,:])/np.sqrt(np.sum((coord_i[2:3,:]-coord_i[1:2,:])**2))
#cos_θ, sin_θ = vec_12[:,0]/np.sqrt(vec_12[:,0]**2 + vec_12[:,1]**2), vec_12[:,1]/np.sqrt(vec_12[:,0]**2 + vec_12[:,1]**2)
#cos_φ, sin_φ = vec_12[:,2], np.sqrt(vec_12[:,0]**2 + vec_12[:,1]**2)
#
#θ = calc_θ(cos_θ, sin_θ)
#θneg = (-θ)*math.pi/180.
#
#φ = calc_θ(cos_φ, sin_φ)
#φneg = (φ-90.)*math.pi/180.
#
#R_z = torch.tensor([[math.cos(θneg), -math.sin(θneg), 0.],
#                    [math.sin(θneg), math.cos(θneg), 0.],
#                    [0., 0., 1.]])
#
#R_y = torch.tensor([[math.cos(φneg),0, -math.sin(φneg)],
#                    [0., 1., 0.],
#                    [math.sin(φneg),0, math.cos(φneg)]])
#
#X_primitive = torch.transpose(torch.tensor(np.float32(coord_i - coord_i[1:2,:])),0,1)
#
#X_middle = torch.matmul(R_y,torch.matmul(R_z,X_primitive))
#coord_i_new = np.transpose(X_middle.numpy())
#vec_10 = (coord_i_new[0:1,:]-coord_i_new[1:2,:])/np.sqrt(np.sum((coord_i_new[0:1,:]-coord_i_new[1:2,:])**2))
#cos_β, sin_β = vec_10[:,1]/np.sqrt(vec_10[:,1]**2 + vec_10[:,2]**2), vec_10[:,2]/np.sqrt(vec_10[:,1]**2 + vec_10[:,2]**2)
#
#β = calc_θ(cos_β, sin_β)
#βneg = (-β)*math.pi/180.
#
#### for later calculation
#θneg, φneg, βneg = -θneg, -φneg, -βneg
#
#Q_list = np.append(Q_list,θneg)
#Q_list = np.append(Q_list, φneg)
#Q_list = np.append(Q_list, βneg)
#Q_list = np.append(Q_list,T_XYZ)
#
#


# In[ ]:




