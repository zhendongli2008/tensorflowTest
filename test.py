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
ova0 = mpslib.mps_dot(mps0,mps0)
print '<p|p>=',ova0

mps = [0]*L
for i in range(L):
   nm = 'mps_site'+str(i)
   if i == 0:
      mps[i] = tf.Variable(mps0[i].reshape(n,1,D),name=nm)
   elif i == L-1:
      mps[i] = tf.Variable(mps0[i].T.reshape(n,D,1),name=nm)
   else:
      mps[i] = tf.Variable(mps0[i].transpose(1,0,2),name=nm)

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
# Also, deriable
mat = tf.reshape(mps[0],[n,D])
lenv[0] = tf.matmul(mat,mat,transpose_a=True)

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

tr0 = tf.reduce_sum(tf.diag_part(lenv[0]))
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

   #
   # Test Slicing:
   #
   # This operation extracts a slice of size size from a tensor input starting at the location specified by begin. The slice size is represented as a tensor shape, where size[i] is the number of elements of the 'i'th dimension of input that you want to slice. 
   print sess.run(mps[0])
   print mps[0].get_shape()
   print sess.run(tf.slice(mps[0],[0,0,0],[1,1,3]))
   print sess.run(tf.slice(mps[0],[1,0,0],[1,1,3]))
   print sess.run(tf.slice(mps[0],[0,0,1],[2,1,1]))
   print sess.run(tf.slice(mps[0],[0,0,1],[2,1,2]))

# Use partial data
mps00 = tf.slice(mps[0],[0,0,0],[1,1,D])
mat = tf.reshape(mps00,[D])
tr1 = tf.reduce_sum(tf.mul(mps00,mps00))
g1 = tf.gradients(tr1,mps[0],name='CurrentGradient')
with tf.Session() as sess:
   sess.run(init)
   print 'mps00',sess.run(mps00)
   print 'tr1',sess.run(tr1)
   print 'g1',sess.run(g1)
   writer = tf.train.SummaryWriter("logs/", sess.graph)

#######
# ova  
####### 
lenv = [0]*L
renv = [0]*L
mat = tf.reshape(mps[0],[n,D])
lenv[0] = tf.matmul(mat,mat,transpose_a=True)
# --   -----
# |  *   |
# --   -----
def leftPropogate(leftEnv,site):
   nshape,lshape,rshape = site.get_shape()
   nshape = nshape.value
   lshape = lshape.value # tf.Dimension
   rshape = rshape.value
   updatedEnv = tf.Variable(tf.zeros([rshape,rshape],dtype=tf.float64))
   for i in range(nshape):
      ti = tf.reshape(tf.slice(site,[i,0,0],[1,lshape,rshape]),[lshape,rshape])
      tmp = tf.matmul(leftEnv,ti)
      tmp = tf.matmul(ti,tmp,transpose_a=True)
      updatedEnv = tf.add(updatedEnv,tmp)
   return updatedEnv

# Define the graph recursively
for i in range(1,L):
   print 'i=',i
   lenv[i] = leftPropogate(lenv[i-1],mps[i])

init = tf.initialize_all_variables()  # must have if define variable
# Try to contract <Psi|Psi>
with tf.Session() as sess:
   sess.run(init)
   res = sess.run(lenv[L-1])
   ova = sess.run(tf.reshape(res,[]))
   print ova
   print ova0

# gradient
# dova = 

# opt?
#


