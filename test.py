#
# The first project with TensorFlow should be to 
# implement an MPS expectation value (e.g. overlap).
#
# Then the energy for the Heisenberg chain (using the MPO operator), 
# then the energy optimization.
#
import tensorflow as tf
import numpy
import mpslib

L = 10
D = 3
n = 2 
numpy.random.seed(0)
mps0 = mpslib.mps_random0(L,n,D)
print '<p|p>=',mpslib.mps_dot(mps0,mps0)

mps = [0]*L
for i in range(L):
   if i == 0:
      mps[i] = tf.Variable(mps0[i])
   elif i == L-1:
      mps[i] = tf.Variable(mps0[i].T)
   else:
      mps[i] = tf.Variable(mps0[i].transpose(1,0,2))

init = tf.initialize_all_variables()  # must have if define variable

# How to do assignment?

# initialize and shape
with tf.Session() as sess:
    sess.run(init)
    sess.run(mps)
    for i in range(L):
       print mps[i].get_shape()

lenv = [0]*L
renv = [0]*L
lenv[0] = tf.matmul(mps[0],mps[0],transpose_a=True)

##### print lenv[0][0]
##### exit()
##### 
##### # --   -----
##### # |  *   |
##### # --   -----
##### def leftPropogate(l0,site):
#####    tf.add(x,y)	
#####    pass

# A Tensor. Has the same type as input. Shape is [M+1, M].
# the first row is eigenvalues, columns of other part are eignvectors.
res = tf.self_adjoint_eig(lenv[0])

import scipy.linalg
l0 = numpy.einsum('pi,pj->ij',mps0[0],mps0[0])
e,v = scipy.linalg.eigh(l0)
print e
print v

flt = tf.reshape(mps[0],[-1])
print 'shp',flt.get_shape()

tr0 = tf.reduce_sum(lenv[0])
g0 = tf.gradients(tr0,mps[0])

tr1 = tf.reduce_sum(tf.mul(mps[0],mps[0]))
g1 = tf.gradients(tr1,mps[0])

with tf.Session() as sess:
   sess.run(init)
   l0 = sess.run(lenv[0])
   #print 'l0',l0
   #print 'res',sess.run(res)
   print 'mps[0]',sess.run(mps[0])
   print 'tr0',sess.run(tr0)
   print 'g0',sess.run(g0)
   print 'tr1',sess.run(tr1)
   print 'g1',sess.run(g1)

# contract <Psi|Psi>

# ova = 

# gradient
# dova = 

# opt?
#

