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

L = 50
D = 10
n = 2 
numpy.random.seed(0)
mps0 = mpslib.mps_random0(L,n,D)
mpslib.mps_normalize(mps0)
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
   #writer = tf.train.SummaryWriter("logs/", sess.graph)

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

ova = tf.reshape(lenv[L-1],[])
# gradient
dOVAdL = tf.gradients(ova,mps[-1],name='dOVAdL')
ic = 4
dOVAdC = tf.gradients(ova,mps[ic],name='dOVAdC')

init = tf.initialize_all_variables()  # must have if define variable
# Try to contract <Psi|Psi>
with tf.Session() as sess:
   sess.run(init)
   print sess.run(ova)
   print ova0

   # Test for d<P|P>/dA[-1]
   print sess.run(dOVAdL)
   v1 = tf.reshape(mps[-1],[n*D])
   v2 = tf.reshape(dOVAdL[0],[n*D])
   # This is correct 
   print sess.run(tf.reduce_sum(tf.mul(v1,v2))/2.0)
 
   # Test for d<P|P>/dA[4]
   print sess.run(dOVAdC)
   v1 = tf.reshape(mps[ic],[n*D*D])
   v2 = tf.reshape(dOVAdC[0],[n*D*D])
   # This is correct 
   print sess.run(tf.reduce_sum(tf.mul(v1,v2))/2.0)
    
   writer = tf.train.SummaryWriter("logs/", sess.graph)

#
# optization for \sum_{all indicies} (A1 - A0)^2 => A1=A0 
#
mps1 = [0]*L
for i in range(L):
   nm = 'mps1_site'+str(i)
   if i == 0:
      mps1[i] = tf.Variable(tf.zeros([n,1,D],dtype=tf.float64),name=nm)
   elif i == L-1:
      mps1[i] = tf.Variable(tf.zeros([n,D,1],dtype=tf.float64),name=nm)
   else:
      mps1[i] = tf.Variable(tf.zeros([n,D,D],dtype=tf.float64),name=nm)

diff = 0.0
for i in range(L):
  diff += tf.reduce_sum(tf.square(tf.sub(mps1[i],mps[i])))

grad = tf.gradients(diff,mps1[0])

mini = tf.train.GradientDescentOptimizer(0.3).minimize(diff,var_list=mps1)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

ifprt = False
for i in range(100):
   #Test gradient: Should be -2A0 if A1==0
   #print 'grad=',sess.run(grad) 
   #print sess.run(mps[0])
   #print sess.run(mps1[0])

   if i == 0:
      print '\nPrint initial data:'
      print '\nsite[0]'
      print 'mps0-ndarray'
      print mps0[0]
      print 'mps-tf'
      print sess.run(mps[0])
      print 'mps1-tf'
      print sess.run(mps1[0])
      
      print '\nsite[-1]'
      print 'mps0-ndarray'
      print mps0[-1]
      print 'mps-tf'
      print sess.run(mps[-1])
      print 'mps1-tf'
      print sess.run(mps1[-1])

   if i % 2 == 0:
      print '\ni=',i,'diff=',sess.run(diff)
      if ifprt:
         print 'before:'
	 print sess.run(mps[0])
         print sess.run(mps1[0])
   # opt
   sess.run(mini)
   if i % 2 == 0:
      if ifprt:
         print 'after:'
	 print sess.run(mps[0])
         print sess.run(mps1[0])
   if sess.run(diff) < 1.e-16:
      print '\nPrint final results:'
      print '\nsite[0]'
      print 'mps0-ndarray'
      print mps0[0]
      print 'mps-tf'
      print sess.run(mps[0])
      print 'mps1-tf'
      print sess.run(mps1[0])
      
      print '\nsite[-1]'
      print 'mps0-ndarray'
      print mps0[-1]
      print 'mps-tf'
      print sess.run(mps[-1])
      print 'mps1-tf'
      print sess.run(mps1[-1])
      break

print tf.trainable_variables()
print mps[0]
print mps[0].name
print mps[0].value()

# 
# tensordot? follow my_qtensor implementation
#
def tf_tensordot(tf1,tf2,axes):
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

init = tf.initialize_all_variables()  # must have if define variable
with tf.Session() as sess: 
   sess.run(init)
   print 'l0'
   print sess.run(lenv[0]) 
   l0 = tf_tensordot(mps[0],mps[0],axes=([0,1],[0,1]))
   print sess.run(l0)

#
# optization for minimize |psi1-psi0|^2 = <1|1> - 2*<0|1> + <0|0>
#
print '\n=============== Least square fit ============'

def tf_mpsgen(L,n,D):
   mps0 = mpslib.mps_random0(L,n,D)
   mpslib.mps_normalize(mps0)
   mps = [0]*L
   for i in range(L):
      nm = 'mps_site'+str(i)
      if i == 0:
         mps[i] = tf.Variable(mps0[i].reshape(n,1,D),name=nm)
      elif i == L-1:
         mps[i] = tf.Variable(mps0[i].T.reshape(n,D,1),name=nm)
      else:
         mps[i] = tf.Variable(mps0[i].transpose(1,0,2),name=nm)
   return mps

def tf_mpsdot(mps1,mps2):
   tmp1 = tf_tensordot(mps1[0],mps2[0],axes=([0,1],[0,1]))
   N = len(mps1)
   for i in range(1,N):
      tmp2 = tf_tensordot(tmp1,mps2[i],axes=([1],[1]))
      tmp1 = tf_tensordot(mps1[i],tmp2,axes=([1,0],[0,1]))
   ova = tf.reshape(tmp1,[])
   return ova

import math
def tf_mpsnormalize(mps):
   norm2 = tf_mpsdot(mps,mps)
   tf_mpsscale(mps,tf.rsqrt(norm2))
   return tf.constant(0)

def tf_mpsscale(mps,alpha):
   N = len(mps)
   fac = tf.pow(alpha,1.0/float(N))
   for i in range(N):
      mps[i] = tf.mul(mps[i],fac)
   return tf.constant(0)

D1 = 20
tf_mpsnormalize(mps)
mps1 = tf_mpsgen(L,n,D1)
normalization = tf.rsqrt(tf.mul(tf_mpsdot(mps1,mps1),tf_mpsdot(mps,mps)))
diff = 2.0-2.0*tf.mul(tf_mpsdot(mps,mps1),normalization)
mini = tf.train.GradientDescentOptimizer(0.3).minimize(diff,var_list=mps1)
#mini = tf.train.MomentumOptimizer(0.1,0.1).minimize(diff,var_list=mps1)
#mini = tf.train.RMSPropOptimizer(0.3).minimize(diff,var_list=mps1)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

nsteps = 100
difflst = []

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.xlim(0,nsteps)
plt.ylim(-0.1,2.1)
plt.xlabel('steps')
plt.ylabel('error')
plt.ion()
plt.show()

for i in range(nsteps):

   diffc = sess.run(diff)
   print '\ni=',i
   print 'n0',sess.run(tf_mpsdot(mps,mps))	
   print 'n1',sess.run(tf_mpsdot(mps1,mps1))	
   print 'df',diffc

   # opt
   sess.run(mini)
   
   # to visualize the result and improvement
   try:
       ax.lines.remove(lines[0])
   except Exception:
       pass
   # plot the prediction
   difflst.append(diffc)
   lines = ax.plot(range(i+1),difflst,'ro-',lw=2)
   plt.pause(0.4)

exit()
