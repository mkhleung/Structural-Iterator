# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:33:51 2019
this script implements optimization of a 80units-tall truss structure subjected to 100units load to vertical height of 80
Optimization is implement by genetic algorithm, where the trait of the best-performing structure is passed to the subsequent generation

@author: mleung
"""
# Third-party libraries
import numpy as np

import copy

import matplotlib.pyplot as plt

# =============================================================================
# 
# =============================================================================

## function initialize model data
def Model(XYZ_input,CON_input,BOUN_input):
    XYZ_shape = np.shape(XYZ_input)
    CON_shape = np.shape(CON_input)
    nn = XYZ_shape[0]        #number of nodes
    ndm = XYZ_shape[1]    #number of dimensions
    XYZ = copy.deepcopy(XYZ_input)            #nodal coordinates array
    ne = CON_shape[0]        #number of elements
    CON = copy.deepcopy(CON_input)            #connectivity array
    nen = np.ones(ne)*2        #number of end nodes of each element
    nq = np.ones(ne)*1         #number of basic forces of each element
    ndf = np.ones(ne)*2        #number of DOFs per node per element
    nt = nn*ndm            #total number of DOFs
    nf = nt-np.sum(BOUN_input) #number of free DOFs
    DOF_vec = np.arange(1,nt+1)    
    DOF = np.reshape(DOF_vec,[XYZ_shape[0],XYZ_shape[1]])

    return [nn, ndm, XYZ, ne, CON, nen,nq,ndf,nt,nf,DOF]

## for the desired element # "el" return the its end node coords in "xyz"    
# for the desired element # "el" return the its end node DOF in "id"    
def Localize(XYZ,DOF,CON,ndf,el):      
    CON = CON[el-1]
    ndf = ndf[el-1]

    xyz = [XYZ[CON[0]-1] , XYZ[CON[1]-1]]           #this works for 2D models only
    xyz = np.transpose(xyz)
    
    id_el = [DOF[CON[0]-1] , DOF[CON[1]-1]]           #this works for 2D models only\
    id_el = np.reshape(id_el,[4,1])
    return [xyz, id_el]

## return length and direction cosine    
def ElmLenOr(xyz):
    Dxyz = xyz[:,1]-xyz[:,0]
    L = np.sqrt(np.dot(Dxyz,Dxyz))
    dcx = np.dot(Dxyz,1/L)
    dcx = np.reshape(dcx,[2,1])
    return [L, dcx]


def ag_matrix(xyz):
    [L, dcx] = ElmLenOr(xyz)
    ### this is for truss only
    ag = np.zeros((1,4))  # #only works for trusses,-dcx, dcx
    ag = np.reshape([-dcx,dcx],(4,1))
    
    return ag

def A_matrix(XYZ,CON,BOUN):
    [nn,ndm,XYZ,ne,CON,nen,nq,ndf,nt,nf,DOF] = Model(XYZ,CON,BOUN)
    k=0
    A = np.zeros((ne,nt))  # 
    for el_count in range(ne):
        [xyz, id_el] = Localize(XYZ,DOF,CON,ndf,el_count+1)
        ag = ag_matrix(xyz)
        #ag_shape = np.shape(ag) #for truss only
        #nc = row(ag)     
        for j in range(len(ag)):    
            
            A[k, id_el[j]-1] = ag[j]
        
        k = k+1 # assumes ag is 1 row only (i.e truss)
    return A
    
def Af_matrix(XYZ,A,BOUN):
        nt = len(XYZ)*2
        nf = nt-np.sum(BOUN)
        BOUN_vec = np.reshape(BOUN,nt)
        dofs_Boun = np.where(BOUN_vec==1)
            
        Af = A
        
        for i in range(nt-nf):
            Af = np.delete(Af,dofs_Boun[0][i]-i,1)
            
        return Af
    
def k_matrix(E,A,L):
    k = E*A/L
    return k

def Ks_matrix(XYZ,CON,BOUN,E,A):
    [nn,ndm,XYZ,ne,CON,nen,nq,ndf,nt,nf,DOF] = Model(XYZ,CON,BOUN)
    for el_count in range(ne):
        [xyz, id_el] = Localize(XYZ,DOF,CON,ndf,el_count+1)
        [L, dcx] = ElmLenOr(xyz)
        if el_count==0:
            k = [E*A[el_count]/L]
        else:
            k.append(E*A[el_count]/L)
    #np.kron(np.eye(2),xyz)
    
    Ks = np.eye(ne)*k
    return [Ks, k]

def Pf_matrix(XYZ,P0,BOUN):
    nt = len(XYZ)*2
    nf = nt-np.sum(BOUN)
    BOUN_vec = np.reshape(BOUN,nt)
    dofs_Boun = np.where(BOUN_vec==1)
        
    Pf = np.reshape(P0,(nt,1))
    
    for i in range(nt-nf):
        Pf = np.delete(Pf,dofs_Boun[0][i]-i,0)
    return Pf

def U_final_(XYZ,BOUN,Uf):
    nt = len(XYZ)*2         # number of total dofs
    BOUN_vec = np.reshape(BOUN,nt)

    dofs_Free = np.where(BOUN_vec==0)   # index for free dofs
    U = np.zeros((nt,1))
    for i in range(len(dofs_Free[0][:])):
        U[dofs_Free[0][i]]=Uf[i]
    return U

def XYZ_final_func(XYZ,U):
    XYZ_len = len(XYZ)
    count=0
    XYZ_final = copy.deepcopy(XYZ)
    for i in range(XYZ_len):
        XYZ_final[i][0]=XYZ_final[i][0]+U[count]
        count=count+1
        XYZ_final[i][1]=XYZ_final[i][1]+U[count]
        count=count+1
    return XYZ_final

def Plot_Model(XYZ,label1,label2,linewid):
    #plot nodes
    for i in range(len(XYZ)):
        plt.show()
        plt.plot(XYZ[i][0],XYZ[i][1],label1)

    #Plot elements
    for i in range(len(CON)):
        plt.show()
        plt.plot([XYZ[CON[i][0]-1][0],XYZ[CON[i][1]-1][0]],[XYZ[CON[i][0]-1][1],XYZ[CON[i][1]-1][1]],label2,linewidth=linewid[i])

def Solve_Model(XYZ,CON,BOUN,P0,E,A_els,mag):
    [nn,ndm,XYZ,ne,CON,nen,nq,ndf,nt,nf,DOF] = Model(XYZ,CON,BOUN)
    #[xyz, id_el] = Localize(XYZ,DOF,CON,ndf,3)
    A =  A_matrix(XYZ,CON,BOUN)
    [Ks, kvec] = Ks_matrix(XYZ,CON,BOUN,E,A_els)
    Af = Af_matrix(XYZ,A,BOUN)
    Kf = np.matmul(np.matmul(np.transpose(Af),Ks),Af)
    K = np.matmul(np.matmul(np.transpose(A),Ks),A)
    Pf = Pf_matrix(XYZ,P0,BOUN)
    Uf = np.matmul(np.linalg.inv(Kf),Pf)
    V = np.matmul(Af,Uf)
    Q = np.matmul(Ks,V)
    #print('Q =', Q)
    U = U_final_(XYZ,BOUN,Uf)
    #print('U =', U)       
    P = np.matmul(K,U)
    #print('P =', P)

    return [V,Q,U,P]

def find_in_vec(vec,a):
    ans = np.where(vec==a)
    ans = ans[0][0]
    return ans

def find_best_parent(vec,num):
    ##### this function somehow messes up the input vector and sorts it
    best=[]
    sortedvec = copy.deepcopy(vec)
    sortedvec = sorted(sortedvec)
    for i in range(num):
        best.append(find_in_vec(vec,sortedvec[len(sortedvec)-i-1]))
    return best

def crossover(matrix1,matrix2,mutationSF):
    size1 = np.shape(matrix1)
    isize = size1[0]
    jsize = size1[1]
    matrix_offspring = np.zeros((isize,jsize))
    for j in range(jsize):    
        for i in range(isize):
            allel = np.random.randn()
#            allel = np.random.randn(1,1)
#            allel = allel[0][0]
            SF = np.random.randn()
#            SF = np.random.randn(1,1)
#            SF = SF[0][0]
            if abs((matrix1[i,j]-matrix2[i,j])/matrix1[i,j]) < 0.10:
                matrix_offspring[i,j]=(matrix1[i,j]+matrix2[i,j])/2*(1+SF*mutationSF) 
            else:
                if allel > 0:
                    matrix_offspring[i,j]=matrix1[i,j]*(1+SF*mutationSF) 
                else:
                    matrix_offspring[i,j]=matrix2[i,j]*(1+SF*mutationSF) 
    return matrix_offspring

# =============================================================================
# INPUTS BELOW
# =============================================================================

## COORDINATES ARRAY
#coordinates define coordinates. # of coord = # of rows 
#XYZ = np.array([[0,0],
#               [8,0],
#               [0,8],
#               [8,8],
#               [0,16],
#               [8,16],
#               [0,24],
#               [8,24],
#               [0,32],
#               [8,32]])
# =============================================================================
# Set up genetic Algorithm
# =============================================================================


num_parents = 4 #choose divisble by 4
choose_parents = 1
epoch = 200

mutationSF_A = 0.05
mutationSF_XYZ = 0.05

Y_max_arr = np.zeros([epoch,num_parents])
objective_arr = np.zeros([epoch,num_parents])
objective_best_ea_epoch = np.zeros([epoch,1])
best_parent_arr = np.zeros([epoch,num_parents])

xi = -8

X = np.transpose([xi,20,
                  8,20,
                  xi,40,
                  8,40,
                  xi,60,
                  8,60])

X = np.reshape(X,[len(X),1])

X_arr = np.zeros([num_parents,len(X),1])

for i in range(num_parents):
    for j in range(len(X)):
        X_arr[i,j,0] = X[j]*(1+np.random.randn()*mutationSF_XYZ)



XYZ_init = np.ones((epoch,num_parents,10,2))

## CONNECTIVITY ARRAY
#define element and the nodes they connect to. # of row = # of elements
CON = np.array( [[1,3],  
                [1,4],
                [1,2],
                [2,3],
                [2,4],
                
                [3,5],
                [3,4],
                [3,6],
                [4,5],
                [4,6],
                
                [5,7],
                [5,6],
                [5,8],
                [6,7],
                [6,8],
                
                [7,9],
                [7,8],
                [7,10],
                [8,9],
                [8,10],
                
                [9,10]])  

# define boundary condition.  "1" = bound, "0" = free
BOUN = np.array([[1,1],
                [1,1],
                [0,0],
                [0,0],
                [0,0],
                [0,0],
                [0,0],
                [0,0],
                [0,0],
                [0,0]])

P0 = np.zeros(np.shape(BOUN))
P0[9,0]= 100    

print('===================================== initiating A_els_arr')
A_els_arr = np.abs(np.random.randn(epoch,num_parents,len(CON),1)*20)
#A_els_arr = np.abs(np.ones((epoch,num_parents,len(CON),1))*20)
    
for k in range(epoch):
    print('===================================== iterating on epoch #:',k)
    for j in range(num_parents):
        #print('=============== iterating on parent #:',j)
        ## define coordinate
        ## intialize parents
        XYZ_basic = np.array([[-8,0],
                               [8,0],
                               [X_arr[j,0,0],X_arr[j,1,0]],
                               [X_arr[j,2,0],X_arr[j,3,0]],
                               [X_arr[j,4,0],X_arr[j,5,0]],
                               [X_arr[j,6,0],X_arr[j,7,0]],
                               [X_arr[j,8,0],X_arr[j,9,0]],
                               [X_arr[j,10,0],X_arr[j,11,0]],
                               [-8,80],
                               [8,80]])

                        
        XYZ_init[k,j,:,:] = XYZ_basic
        
        XYZ_init_shape = np.shape(XYZ_init)
        
        XYZ = copy.deepcopy(XYZ_basic)
        
        ## define member areas
        A_els = A_els_arr[k,j,:,0]
        A_els = np.reshape(A_els,(len(A_els),1))
        
        ## determine highest coordination
        Y_max = max(XYZ[:,1])
        Y_min = min(XYZ[:,1])
        Y_max_arr[k,j] = Y_max
        Y_max_where = np.where(XYZ==Y_max) 
        
        ## define elastic modulus        
        E = 29000
     
        ## calculate results
        [V,Q,U,P] = Solve_Model(XYZ,CON,BOUN,P0,E,A_els,100)
        
        stress = np.abs(Q/A_els)
        
        Amin = np.abs(A_els*stress)/36
        
        ## calculate member lengths
        L_els = np.zeros((len(CON),1))
        for i in range(len(CON)):
            [nn,ndm,XYZ,ne,CON,nen,nq,ndf,nt,nf,DOF] = Model(XYZ,CON,BOUN)
            [xyz, id_el] = Localize(XYZ,DOF,CON,ndf,i+1)
            [L, dcx] = ElmLenOr(xyz)
            L_els[i] = L
        
        # =============================================================================
        # 
        # =============================================================================
        cost = np.dot(np.transpose(L_els),(Amin+(A_els-Amin)**2))
        
        objective_arr[k,j] = np.sign(Y_max)*(Y_max-Y_min)/cost
        
        

    best_parent = find_best_parent(objective_arr[k,:],num_parents)
    best_parent_arr[k,:] = best_parent
    objective_best_ea_epoch[k,0] = objective_arr[k,best_parent[0]]
    print('objective = ',objective_arr[k,best_parent[0]])
    
    #=========================== plot force results
    plt.clf()
    plt.figure(1)
#    Plot_Model(XYZ,'ro','r-',Q*0.05)
    
    #=========================== plot displacement results
#    XYZ_magnified = XYZ_final_func(XYZ_init[best_parent[0],:,:],U*10)
#    plt.figure(1)
#    plt.xlim([0,10])
#    plt.ylim([0,100])
    Plot_Model(XYZ_init[k,best_parent[0],:,:],'bo','b-',A_els*0.25)
    plt.show()
#    Plot_Model(XYZ_magnified,'ro','r--',A_els*0.1)     
    plt.pause(0.01)
    
    #=========================== revise weights and biases of each offspring
    # initialize zero matrix to write new bias/weight information
    XYZ_init_n = np.zeros((num_parents,XYZ_init_shape[2],2))
    A_els_arr_n = np.zeros((num_parents,len(CON),1))
    
    
    for i1 in range(num_parents):
        #print('=============== crossover on parent #:',i1)
        j1 = round(i1/2)
        
        X_n = crossover(X_arr[best_parent[j1-1],:,:],X_arr[best_parent[choose_parents-j1-1],:,:],mutationSF_XYZ)
        A_els_n = crossover(A_els_arr[k,best_parent[j1-1],:,:],A_els_arr[k,best_parent[choose_parents-j1-1],:,:],mutationSF_A)
        
        ### replace previous parent array with new parents array
        X_arr[i1,:,:] = X_n      
        A_els_arr_n[i1,:,:] = A_els_n        
        
    if k == epoch-1:
        print('run ends')
    else:
        A_els_arr[k+1,:,:,:] = A_els_arr_n

[V,Q,U,P] = Solve_Model(XYZ_init[epoch-1,best_parent[0],:,:],CON,BOUN,P0,E,A_els_arr[epoch-1,best_parent[0],:,:],1)
plt.clf()
plt.figure(1)
plt.show()
Plot_Model(XYZ_init[epoch-1,best_parent[0],:,:],'bo','b-',A_els_arr[epoch-1,best_parent[0],:,:]*1)
Plot_Model(XYZ_init[epoch-1,best_parent[0],:,:],'ro','r--',abs(Q)*0.05)

(abs(Q/36)-A_els_arr[epoch-1,best_parent[0],:,:])

plt.figure(2)
plt.plot(np.arange(0,epoch),objective_best_ea_epoch)
