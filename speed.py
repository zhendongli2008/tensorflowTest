import tensorflow as tf
import tflib
import time
import matplotlib.pyplot as plt

L = 30
n = 4
D1 = 600
mps1 = tflib.mps_rand(L,n,D1)

ova = tflib.mps_dot(mps1,mps1)

g = [0]*L
for i in range(L-1,-1,-1):
   g[i] = tf.gradients(ova,mps1[i],name='g'+str(i))

fig = plt.figure()
   
init = tf.initialize_all_variables()
with tf.Session() as sess:

   sess.run(init)

   t0 = time.time()
   print sess.run(ova)
   t1 = time.time()
   dt0 = t1-t0
   print 't[ova]=',dt0
   
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
