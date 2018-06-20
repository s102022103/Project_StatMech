import pyuni10 as uni10
import numpy as np
import matplotlib as plt

def TRG(beta,chi=16,times=15):
    Delta = matD()
    M     = matM(beta)
    M_T   = matMT(beta)
    T     = merge(Delta,M,M_T)
    for iteration in range(times):
        T2 = update(T,chi)
        T  = update(T2,chi)
    return tTrace(T,T,T,T)

def matD():
    """create delta tensor"""
    mat = np.zeros(16)
    mat[0]=mat[15]=1
    bdout = uni10.Bond(uni10.BD_OUT,2)
    T   = uni10.UniTensorR([bdout,bdout,bdout,bdout])
    T.SetElem(mat)
    return T

def matM(beta):
    mat = np.array([[np.cosh(beta)**(1/2),np.sinh(beta)**(1/2)],
                    [np.cosh(beta)**(1/2),-1*np.sinh(beta)**(1/2)]])
    T   = uni10.UniTensorR(mat)
    return T

def matMT(beta):
    mat  = matM(beta).GetBlock()
    T    = uni10.UniTensorR(np.transpose(mat))
    return T

def merge(delta,m,mt):
    """ initially the tensors are different, so I decide to merge M/MT into delta tensor such that
    the tensors are the same, therefore we only need to consider 1 tensor when doing TRG"""
    network = uni10.Network("merge_bond.net")
    result  = uni10.UniTensorR(delta)
    network.PutTensor('D',delta)
    network.PutTensor("M",m)
    network.PutTensor("MT",mt)
    network.Launch(result)
    return result

def update(T,chi):
    T2   = CG_contraction(T,T)
    copy = uni10.UniTensorR(T2)
    copy.SetLabel([3,4,1,-2])
    X    = uni10.Contract(T2,copy)
    U,norm    =  svd(X.GetBlock(),chi)  # use SVD to truncate the tensor with bond dimension chi
    result = iso_contract(T2,U)
    norm   = np.max(result.GetBlock())/2
    result = uni10.Permute(result,[2,1,4,3],2)
    return result*(1/norm)

def CG_contraction(A,B):
    net    = uni10.Network("CG_contract.net")
    result = uni10.UniTensorR(A)
    net.PutTensor(0,A)
    net.PutTensor(1,B)
    net.Launch(result)
    result = result.CombineBond([2,-2])
    result = result.CombineBond([4,-4])
    return result

def svd(npmat,bd,threshold=10**-8):
    """calling svd in numpy and truncate the matrix(npmat) with bond dimension (at most bd)"""
    u,s,vd = np.linalg.svd(npmat)
    bd2    = np.min([np.sum(s>=threshold),bd])
    u      = u[:,:bd2]
    s      = s[:bd2]
    return u,s

def iso_contract(T,isoT):
    """ applying isometry and its transpose on tensor T, shrinkage T's dimension"""
    iso_net = uni10.Network('isometry.net')
    res    = uni10.UniTensorR(T)
    isoT   = uni10.UniTensorR(isoT)
    T_trans= uni10.UniTensorR(np.transpose(isoT.GetBlock()))
    iso_net.PutTensor(0,T)
    iso_net.PutTensor(1,isoT)
    iso_net.PutTensor(2,T_trans)
    iso_net.Launch(res)
    return res

def tTrace(Ta,Tb,Tc,Td):
    """ return the tensor trace result"""
    net = uni10.Network("trace.net")
    res = uni10.UniTensorR(Ta)
    net.PutTensor(0,Ta)
    net.PutTensor(1,Tb)
    net.PutTensor(2,Tc)
    net.PutTensor(3,Td)
    net.Launch(res)
    scalar = res.GetBlock()[0]
    return scalar

print(TRG(4.1,chi=10,times=10))