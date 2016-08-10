import tensorflow as tf
import tflib
import time
import matplotlib.pyplot as plt

L = 20
n = 4
D1 = 200
mps1 = tflib.mps_rand(L,n,D1)

ova = tflib.mps_dot(mps1,mps1)
ovaLR = tflib.mps_dotLR(mps1,mps1)

g = [0]*L
for i in range(L):
   g[i] = tf.gradients(ova,mps1[i],name='g'+str(i))

gLR = [0]*L
for i in range(L):
   gLR[i] = tf.gradients(ova,mps1[i],name='gLR'+str(i))

fig = plt.figure()
  
fg = tf.gradients(ova,mps1,name='fg')
fgLR = tf.gradients(ovaLR,mps1,name='fgLR')

init = tf.initialize_all_variables()
print 'start session'
with tf.Session() as sess:

   t0 = time.time()
   sess.run(init)
   t1 = time.time()
   dt0 = t1-t0
   print 't[init]=',dt0

   t0 = time.time()
   print sess.run(ova)
   t1 = time.time()
   dt0 = t1-t0
   print 't[ova]=',dt0
 
   t0 = time.time()
   print sess.run(ovaLR)
   t1 = time.time()
   dt0LR = t1-t0
   print 't[ovaLR]=',dt0LR
 
   t0 = time.time()
   sess.run(fg)
   t1 = time.time()
   dt1 = (t1-t0)/dt0
   print 'r[fg]=',dt1
 
   t0 = time.time()
   sess.run(fgLR)
   t1 = time.time()
   dt1LR = (t1-t0)/dt0
   print 'r[fgLR]=',dt1LR
  
   tlst = [0]*L
   tlstLR = [0]*L
   for i in range(L):
      t0 = time.time()
      sess.run(g[i])
      t1 = time.time()
      dt = (t1-t0)/dt0
      print 'i=',i,'rg[i]=',dt
      print len(g[i]),g[i]
      tlst[i] = dt
  
      t0 = time.time()
      sess.run(gLR[i])
      t1 = time.time()
      dt = (t1-t0)/dt0
      print 'i=',i,'rgLR[i]=',dt
      print len(gLR[i]),gLR[i]
      tlstLR[i] = dt
 
   plt.plot(range(L),tlst,'ro-',lw=2)
   plt.plot(range(L),tlstLR,'bo-',lw=2)
   plt.show()
   print tlst
       
   writer = tf.train.SummaryWriter("logs/", sess.graph)
