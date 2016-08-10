import tensorflow as tf
import tflib
import time
import matplotlib.pyplot as plt

L = 10
n = 4
D1 = 500
mps1 = tflib.mps_rand(L,n,D1)

ova = tflib.mps_dot(mps1,mps1)

g = [0]*L
for i in range(L-1,-1,-1):
   g[i] = tf.gradients(ova,mps1[i],name='g'+str(i))

fig = plt.figure()
  
fg = tf.gradients(ova,mps1,name='fg')

# just for test
N = len(mps1)
tmp = [0]*N
tmp[0] = tflib.tensordot(mps1[0],mps1[0],axes=([0,1],[0,1]))
for i in range(1,N):
   tmp2 = tflib.tensordot(tmp[i-1],mps1[i],axes=([1],[0]))
   tmp[i] = tflib.tensordot(mps1[i],tmp2,axes=([0,1],[0,1]))
ova0 = tf.reshape(tmp[N-1],[],name='ova0')

init = tf.initialize_all_variables()
with tf.Session() as sess:

   sess.run(init)

   t0 = time.time()
   sess.run(ova)
   t1 = time.time()
   dt0 = t1-t0
   print 't[ova]=',dt0
 
   t0 = time.time()
   sess.run(ova0)
   t1 = time.time()
   dt2 = (t1-t0)/dt0
   print 'r[ova0]=',dt2
   
   t0 = time.time()
   sess.run(fg)
   t1 = time.time()
   dt1 = (t1-t0)/dt0
   print 'r[fg]=',dt1
  
   tlst = [0]*L
   for i in range(L):
      t0 = time.time()
      sess.run(g[i])
      t1 = time.time()
      dt = (t1-t0)/dt0
      print 'i=',i,'g[i]=',dt
      tlst[i] = dt
  
   plt.plot(range(L),tlst,'ro-',lw=2)
   plt.show()
   print tlst
       
   writer = tf.train.SummaryWriter("logs/", sess.graph)
