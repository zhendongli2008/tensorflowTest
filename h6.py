# 
# Electronic energy of H6 at R=1A
# FCI: -7.83990801
# HF:  -7.73937394
# D=10:-7.83308985
#
import h5py
import numpy
import tensorflow as tf

def testHmpo(fname='hop',debug=True):
   f = h5py.File(fname)
   nsite = f['nsite'].value
   nops  = f['nops'].value
   hpwts = f['wts'].value
   if debug:
      print '[testHmpo]'
      print 'nsite=',nsite
      print 'nops =',nops
      print 'hpwts=',hpwts
   for isite in range(nsite):  
      gname = 'site'+str(isite)
      grp = f[gname]
      print
      for iop in range(nops):
	 print 'isite,iop,shape=',isite,iop,grp['op'+str(iop)].shape
   f.close()
   return 0

def calHexp(mps,fname='hop',debug=True):
   if debug: print '\n[calHexp]'
   f = h5py.File(fname)
   nsite = f['nsite'].value
   nops  = f['nops'].value
   hpwts = f['wts'].value
   # Lop over sites
   lops = [[0]*nops]*(nsite+1)
   # Boundary
   lop0 = [tf.constant(1.0,dtype=tf.float64,shape=(1,1,1))]*nops
   lops[0] = lop0
   Hpp = tf.Variable(0.,dtype=tf.float64)
   for isite in range(nsite): 
      if debug: print '\n>>> isite =',isite,'<<<'
      gname = 'site'+str(isite)
      grp = f[gname]
      bsite = mps[isite]
      ksite = mps[isite]
      for iop in range(nops):
	 if debug: print 'isite,iop,shape=',isite,iop,grp['op'+str(iop)].shape
	 cop = tf.constant(grp['op'+str(iop)].value)
	 tmp = lops[isite][iop]
	 tmp = tflib.tensordot(bsite,tmp,axes=([0],[1]))
	 tmp = tflib.tensordot(cop,tmp,axes=([0,2],[2,0]))
	 tmp = tflib.tensordot(tmp,ksite,axes=([1,3],[1,0]))
	 lops[isite+1][iop] = tmp
	 if isite == nsite-1: 
	    Hpp += tf.reshape(tf.mul(tmp,hpwts[iop]),[])
   f.close()
   return Hpp 

def optimize(arg,optVal,nsteps=100):
   mini = tf.train.GradientDescentOptimizer(0.3).minimize(optVal,var_list=arg)
   difflst = []
   import matplotlib.pyplot as plt
   fig = plt.figure()
   ax = fig.add_subplot(1,1,1)
   plt.xlim(0,nsteps)
   #plt.ylim(-0.1,2.1)
   plt.xlabel('steps')
   plt.ylabel('error')
   plt.ion()
   plt.show()
   # start
   for i in range(nsteps):
      valc = sess.run(optVal)
      print '\ni=',i,'valc=',valc,'norm=',sess.run(tflib.mps_dot(mps1,mps1))
      # opt
      sess.run(mini)
      # to visualize the result and improvement
      try:
          ax.lines.remove(lines[0])
      except Exception:
          pass
      # plot the prediction
      difflst.append(valc)
      lines = ax.plot(range(i+1),difflst,'ro-',lw=2)
      plt.pause(0.4)
   return valc

if __name__ == '__main__':
   
   testHmpo()

   import tflib
   L = 6
   n = 4
   D1 = 10
   #mps1 = tflib.mps_rand(L,n,D1)
   occun = [1]*6+[0]*6 
   mps1 = tflib.mps_rand0(L,n,D1,occun)

   normalization = tflib.mps_dot(mps1,mps1)
   Hpp = calHexp(mps1)
   energy = tf.div(Hpp,normalization)

   init = tf.initialize_all_variables()
   
   with tf.Session() as sess:
      sess.run(init)
      optimize(mps1,energy,nsteps=300)