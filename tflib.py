#
# The first project with TensorFlow should be to 
# implement an MPS expectation value (e.g. overlap).
#
# Then the energy for the Heisenberg chain (using the MPO operator), 
# then the energy optimization.
#
import tensorflow as tf
import numpy
import math
import mpslib

# 
# tensordot? follow my_qtensor implementation
#
def tensordot(tf1,tf2,axes):
   debug = True
   shp1 = tf1.get_shape()
   shp2 = tf2.get_shape()
   r1 = range(len(shp1))
   r2 = range(len(shp2))
   # Indices
   i1,i2 = axes
   e1 = list(set(r1)-set(i1))
   e2 = list(set(r2)-set(i2))
   ne1 = len(e1)
   ne2 = len(e2)
   nii = len(i1)
   rank = ne1+ne2
   sdx1 = e1+i1 # sort index
   sdx2 = i2+e2
   # Shapes - get_reshape() return Dimensions
   eshp1 = [shp1[i].value for i in e1]
   ishp1 = [shp1[i].value for i in i1]
   eshp2 = [shp2[i].value for i in e2]
   ishp2 = [shp2[i].value for i in i2]
   esize1 = numpy.prod(eshp1) 
   isize1 = numpy.prod(ishp1)
   esize2 = numpy.prod(eshp2)
   isize2 = numpy.prod(ishp2)
   mtf1 = tf.reshape(tf.transpose(tf1,perm=sdx1),[esize1,isize1])
   mtf2 = tf.reshape(tf.transpose(tf2,perm=sdx2),[isize2,esize2])
   tfc = tf.reshape( tf.matmul(mtf1,mtf2) , eshp1+eshp2 )
   return tfc

def mps_rand(L,n,D):
   mps0 = mpslib.mps_random0(L,n,D)
   mpslib.mps_normalize(mps0)
   mps = [0]*L
   for i in range(L):
      nm = 'mps_site'+str(i)
      if i == 0:
         mps[i] = tf.Variable(mps0[i].reshape(1,n,D),name=nm)
      elif i == L-1:
         mps[i] = tf.Variable(mps0[i].reshape(D,n,1),name=nm)
      else:
         mps[i] = tf.Variable(mps0[i],name=nm)
   return mps

def mps_normalize(mps):
   norm2 = mpsdot(mps,mps)
   mpsscale(mps,tf.rsqrt(norm2))
   return tf.constant(0)

def mps_scale(mps,alpha):
   N = len(mps)
   fac = tf.pow(alpha,1.0/float(N))
   for i in range(N):
      mps[i] = tf.mul(mps[i],fac)
   return tf.constant(0)

def mps_dot(mps1,mps2):
   tmp1 = tensordot(mps1[0],mps2[0],axes=([0,1],[0,1]))
   N = len(mps1)
   for i in range(1,N):
      tmp2 = tensordot(tmp1,mps2[i],axes=([1],[0]))
      tmp1 = tensordot(mps1[i],tmp2,axes=([0,1],[0,1]))
   ova = tf.reshape(tmp1,[])
   return ova


def mps_rand0(L,n,D,occun,noise=0.1):
   assert n == 4
   mps0 = mpslib.mps_random0(L,n,D)
   mpslib.mps_normalize(mps0)
   mps = [0]*L
   for i in range(L):
      nm = 'mps_site'+str(i)
      if i == 0:
         site = mps0[i].reshape(1,n,D)
      elif i == L-1:
         site = mps0[i].reshape(D,n,1)
      else:
         site = mps0[i].copy()
      # occ
      na = occun[2*i]
      nb = occun[2*i+1]
      th = 1.e-4
      site = noise*site
      if abs(na)<th and abs(nb)<th:
	 site[0,0,0] = 1.0
      elif abs(na)<th and abs(nb-1.0)<th:
	 site[0,1,0] = 1.0
      elif abs(na-1.0)<th and abs(nb)<th:
	 site[0,2,0] = 1.0
      elif abs(na-1.0)<th and abs(nb-1.0)<th:
	 site[0,3,0] = 1.0
      mps[i] = tf.Variable(site)
   return mps
