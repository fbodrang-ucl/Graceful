# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 14:35:57 2012

@author: Olivier
"""
from scipy.optimize import fminbound    

from os import chdir
from struct import *
import numpy as np
import random as ra
from numpy import *
from numpy import fft as fft
import matplotlib.pyplot as plt
from numpy.linalg import *
from numpy.random import *
from scipy.interpolate import interp1d
from scipy.linalg import *
from scipy.io import *
from scipy.signal import lfilter
from scipy.optimize import fmin

def eof(t,X,n):
  l=len(t)
  s=X.shape
  if s[0]<s[1]:
    X=transpose(X);
  S=transpose(X)@X;
  w, v = eig(S)
  pc=transpose(v)@transpose(X)
  if pc.shape[0] != len(t):
      dum=v
      v=pc
      pc=dum    
  pc=pc[:,:n]
  v=v[:n,:].T    
  return v,pc,w

def demean(x):
   x=x-mean(x)
   return x


def ARpar(X):
   Xb=mean(X)
   N=X.shape[0]
   n=12
   c1=1./(N-1)*dot(squeeze(X[0:-1]-Xb),squeeze(X[1:]-Xb))
   c0=1./N*dot(squeeze(X-Xb),squeeze(X-Xb))
   gh=c1/c0
   for i in range(0,n):
       mu2=-1./N+2./N**2*((N-gh**N)/(1-gh)-gh*(1-gh**(N-1))/(1-gh)**2)
       dg=-(gh-mu2)/(1-mu2)+c1/c0;
       gh=gh+dg;
   c0=c0/(1-mu2);
   ah=sqrt(c0*(1-gh**2))
   return ah,gh
   
def compPC(X,V,N=[0,1,2,3,4,5,6,7,8]):
    M=V.shape[0]
    L=X.size-M+1
    Vn=V[:,N]
    X=demean(X)
    pc=zeros((L,Vn.shape[1]))
    for i in range(0,Vn.shape[1]):
        for k in range(0,L):
           pc[k,i]=dot(squeeze(X[k:k+M]),transpose(Vn[:,i]))
    return pc     
    
def compRC(X,V,N=[0,1,2,3,4,5,6,7,8]):
    pc=compPC(X,V,N)
    X=X-mean(X)
    M=V.shape[1]
    Vn=V[:,N]  
    L=X.size
    rc=zeros((L,Vn.shape[1]))
    for i in range(Vn.shape[1]):
        rc[0,i]=pc[0,i]*Vn[0,i]
        for k in range(1,M):
            rc[k,i]=1./(k+1)*dot(transpose(flipud(pc[0:k+1,i])),Vn[0:k+1,i])
        for k in range(M,L-M+1):
            rc[k,i]=1./M*dot(transpose(flipud(pc[k-M:k,i])),Vn[0:M+1,i])
        for k in range(L-M+1,L): 
            rc[k,i]=1./(L-k)*dot(transpose(flipud(pc[k-M+1:,i])),Vn[k-L+M:,i])   
    return rc        
    
    
def randn(mu,sigma,L,N):
   r=zeros((L,N))
   for j in range(L):
      for i in range(N):
         rn = ra.normalvariate(mu,sigma)
         r[j,i]=rn;   
   return r 

def ar1gen(a,g,L,N):
    x=zeros((L,N));
    z=a*randn(0,1,L,N);
    x[0,:]=z[0,:]
    p=zeros((N))
    for l in range(L):
       x[l,:]=g*x[l-1,:]+z[l,:];
    for l in range(N):
       a,g=ARpar(x[:,l]) 
       p[l]=a
    return x 

    
def objfun(f,t,X):
    A=transpose(vstack([0*t+1, t, cos(2*pi*f*t), sin(2*pi*f*t)]))  
    y,re,s,k0=lstsq(A,X)
    if not re:
       re=var(X)
    return re
    
def mcssa(X,M=0,N=100,co='VG'):
   X=demean(X)
   if M==0:
      M=int(X.size*1./3)
   Ds=zeros((M,N))
   V,D,pc,r=ssa(X,M,co)
   F=zeros((M))
   t=arange(M)
   freq=arange(0,fix(M/2))*1./M;
   for k in range(M):
       fx=fft.fft(V[:,k])
       ik=argmax(abs(fx[0:round(0.5*M)-1]))
       ft=freq[ik]
       p=fmin(objfun, ft, args=(t,V[:,k]),full_output=0,disp=0)
       F[k]=abs(p[0])
       F[k]=ft
   if len(X.shape)==1:
       ah,gh=ARpar(X)
       L=X.size
       xs=ar1gen(ah,gh,L,N)  
       for i in range(N):
           ci=covaBH(demean(squeeze(xs[:,i])),M)
           Ds[:,i]=diag(V.T@ci@V)
   else:
       lx=X.shape[0]
       ah=zeros(lx)
       gh=zeros(lx)
       xs=zeros((M,lx,N))
       for k in range(lx):
           ah[k],gh[k]=ARpar(X[:,k])
           xs[:,:,k]=ar1gen(ah[k],gh[k],lx,N)    
       for i in range(N):
           ci=covaBH(demean(squeeze(xs[:,:,i])),M)
           Ds[:,i]=diag(V.T@ci@V)
   Ds=sort(Ds,axis=1)   
   return D,Ds,F   
   
def mcssaplot(F,D,Ds):  
   M=F.size; 
   N=Ds.shape[1]
   plt.semilogy(F,D,'ro')
   d=mean(diff(F))/2
   for i in range(M):
      plt.semilogy([F[i],F[i]],[Ds[i,round(0.025*N)],Ds[i,round(0.975*N)]],'k')
#      plt.semilogy([F[i]-d,F[i]+d],[Ds[i,round(0.025*N)],Ds[i,round(0.025*N)]],'k')
#      plt.semilogy([F[i]-d,F[i]+d],[Ds[i,round(0.975*N)],Ds[i,round(0.975*N)]],'k')
      plt.axis([-10*d,.5+d,amin(Ds),amax(Ds)])
   plt.show()
       
       
    
def covaVG(X,M=0):
   N=X.size
   if M==0:
      M=int(N*1./3)
   C=zeros((M))
   X=demean(X)
   C[0]=sum(X**2)/N
   for k in range(1,M):
       C[k]=sum(X[0:-k]*X[k:])/(N-k)
   c=toeplitz(C)
   return c
   
   
def covaxy(X,Y,M=0):
   N=X.shape[0];
   if M==0:
      M=int(N*1./3)
   nfft=int(2**(int(log(N-1)/log(2))+2));
   x=fft.fft(X,nfft)
   y=fft.fft(Y,nfft)
   c=fft.ifft(y*conj(x));
   c=real(fft.fftshift(c)); 
   di=zeros((2*M+1))
   di[0:M+1]=arange(N-M,N+1)
   di[M+1:2*M+1]=flipud(arange(N-M,N));
   c=flipud(c[nfft/2-M:nfft/2+M+1])/di;
   return c
   
def covam(X,M=0):
   N=X.shape[0]
   L=X.shape[1]
   if M==0:
      M=int(N*1./3)
   c=zeros((2*M+1,L**2));
   nfft=int(2**(int(log(N-1)/log(2))+2));
   di=zeros((1,2*M+1))
   xtmp=zeros((nfft,L),dtype=complex128);
   for i in range(0,L):
     xtmp[:,i]=fft.fft(X[:,i],nfft)
   ind=arange(nfft/2-M,nfft/2+M+1).astype(int);
   for i in range(0,L):
      for j in range(i,L):
         y=fft.ifft(xtmp[:,j]*conj(xtmp[:,i])); 
         y=real(fft.fftshift(y));
         col=i*L+j;
         c[:,col]=y[ind];
   ind=flipud(arange(0,2*M+1));
   for i in range(1,L):
      for j in range(0,i):
          col=i*L+j;
          colnew=j*L+i;
          c[:,col]=c[ind,colnew];     
   di[0,0:M+1]=arange(N-M,N+1)
   di[0,M+1:2*M+2]=flipud(arange(N-M,N));
   c=c/tile(transpose(di),(1,L**2));
   return c       

def mat_mssa(X,M=0):
    L=X.shape[1]
    if M==0:
        M=int(X.shape[0]/3)
    c=covam(X,M-1);
    T=zeros((L*M,L*M))    
    for k in range(0,L):
        q=c[M-1:2*M-1,k*(L+1)];
        Tij=toeplitz(q);  
        T[k*M:(k+1)*M,k*M:(k+1)*M]=Tij;
        for j in range(k+1,L):
            q=c[:,k*L+j];  
            Tij=toeplitz(q[0:M][::-1],q[M-1:2*M-1]);
            #if (k==0 and j==1):
                #print q[0:M][::-1]
            T[k*M:(k+1)*M,j*M:(j+1)*M]=Tij;
            T[j*M:(j+1)*M,k*M:(k+1)*M]=transpose(Tij)
    return T
            
def compPCm(X,M,V,Nc=[0,1,2,3,4,5,6,7]):
    S=X.shape;
    T=V.shape;
    pc=zeros((S[0],len(Nc)));
    
    a=zeros((S[0],S[1]));
    Ej=zeros((M,S[1]));
    for k in range(0,len(Nc)):
#        print(k,V.shape,S ,Nc)
        Ej=transpose(fliplr(reshape(V[:,Nc[k]],(S[1],M))))
        for j in range(0,S[1]):
           a[:,j]=lfilter(Ej[:,j],1,X[:,j]);
        if S[1]>1:    
          pc[:,k]=sum(a,1)   
        else:
          pc[:,k]=squeeze(a)
    pc=pc[M-1:X.shape[0],:]                   
    return pc     

def compRCm(X,M,V,A,Nc=[0,1,2,3,4,5,6,7]):
    V=V[:,Nc]
    ml,k=V.shape
    ra,ka=A.shape
    L=X.shape[1]
    M=int(ml/L)
    N=ra+M-1
    R=zeros((N,L*len(Nc)))
    Z=zeros((M-1,k))
    A=transpose(hstack((transpose(A),transpose(Z))))
    for j in range(0,len(Nc)):
        Ej=transpose(reshape(V[:,j],(L,M)))
        for i in range(0,L):
            R[:,j*L+i]=lfilter(Ej[:,i],M,A[:,Nc[j]])
    for i in range(0,M-1):
        R[i,:]=R[i,:]*M/(i+1)
        R[N-i-1,:]=R[N-i-1,:]*M/(i+1)
    return R

def mssa(X,M=0,Nc=8):
    N=X.shape[0]
    Nc=range(0,Nc)
    if M==0:
       M=int(N*1./3)
    T=mat_mssa(X,M)
    D,V=eig(T)
    ks=flipud(argsort(D))
    V=V[:,ks]
    D=real(D[ks])
    pc=compPCm(X,M,V,Nc=Nc)
    R=compRCm(X,M,V,pc,Nc=Nc)
    return  V,D,pc,R       
   
def covaBH(X,M=0):
   N=X.size
   if M==0:
      M=int(N*1./3)
   Np=N-M+1
   D=zeros((Np,M))
   X=demean(X)
   for k in range(0,Np):
       D[k,:]=squeeze(X[k:k+M])
   c=dot(transpose(D),D)/Np
   return c
   
def ssa(X,M=0,co='VG',Nc=8):
    N=X.size
    if M==0:
      M=int(N*1./3)
    if co=='VG':
       c=covaVG(X,M)
    elif co=='BH':
       c=covaBH(X,M)
    D,V=eig(c)
    ks=flipud(argsort(D))
    V=V[:,ks]
    D=real(D[ks])
    Nc=range(0,Nc+1)
    X=reshape(X,(N,1))
    pc=compPCm(X,M,V,Nc=Nc)
    R=compRCm(X,M,V,pc,Nc=Nc)
    return  V,D,pc,R
    
    
def ssa_errbar(X):
    N=X.size
    kappa=1.5
    r=xcorr(X,1)
    tau=-1./log(r[1]);
    s=sqrt(2/(N*1./kappa/tau))
    return s
    
def BT_correl(X,M=0,w=0,n=100):
    N=X.size
    if M==0:
      M=int(N*1./5)
    r=xcorr(X,M)
    n=r.size
    r=hstack([flipud(r[1:]),r])
    j=arange(-n+1,n)
    w=windo(r.size,1)
    freq=linspace(0,0.5,N)
    S=zeros(freq.shape)
    for k in range(N):
        S[k]=dot(w*r,exp(-2*1j*pi*freq[k]*j))**2/M
    plt.semilogy(freq,S)
    plt.show()

def windo(N,w):
   W=ones((N,))
   if w==1:
     for k in range(N):
         N2=0.5*(N-1)
         W[k]=(N2-abs(k-N2))/N2
   elif w==2:
     for k in range(N):
        W[k]=0.5*(1.-cos(2*pi*k*1./(N-1)))
   elif w==3:
     for k in range(N):
        W[k]=0.54-0.46*cos(2*pi*k*1./(N-1))
   return w  
    
def xcorr(X,L):
   X=demean(X)
   xc=zeros((L+1))
   xc[0]=1;
   for k in range(1,L+1):
       xc[k]=sum(X[0:-k]*X[k:])/sqrt(sum(X[0:-k]**2)*sum(X[k:]**2))   
   return xc 
   
def xcorr2(X,Y,L):
   X=demean(X)
   Y=demean(Y)
   xc=zeros(2*L+1)
   xc[0]=1;
   xc[L]=sum(X*Y)/sqrt(sum(X**2)*sum(Y**2))
   for k in range(1,L+1):
       xc[L+k]=sum(X[0:-k]*Y[k:])/sqrt(sum(X[0:-k]**2)*sum(Y[k:]**2))  
       xc[L-k]=sum(Y[0:-k]*X[k:])/sqrt(sum(Y[0:-k]**2)*sum(X[k:]**2))
        
   return xc 
   
   