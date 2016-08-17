# 
# Electronic energy of H8 at R=1A
#
import copy
import math
import h5py
import numpy
import tensorflow as tf
import anim
import tflib
import treelib

# hmpo = \sum_{x} wts,Hx with Hx[MPO] is a list of sites
def loadHmpo(fname='hop8',debug=True):
   f = h5py.File(fname)
   nsite = f['nsite'].value
   nops  = f['nops'].value
   hpwts = f['wts'].value
   if debug:
      print '[loadHmpo]'
      print 'nsite=',nsite
      print 'nops =',nops
      print 'hpwts=',hpwts
   #
   # For some reason, mpos = [[0]*nsite]*nops does not work!
   #
   mpos = [0]*nops
   for iop in range(nops):
      lst = [0]*nsite
      for isite in range(nsite):  
         lst[isite] = f['site'+str(isite)+'/op'+str(iop)].value
      mpos[iop] = lst
   f.close()
   hmpo = (hpwts,mpos)
   return hmpo

# optimizer
def optimize(arg,optVal,nsteps=100,ifgif=False):
   depth,sites,hmpo = arg
   # best opt. tested.
   mini = tf.train.AdamOptimizer(0.01).minimize(optVal,var_list=sites)
   # plot
   difflst = []
   import matplotlib.pyplot as plt
   fig = plt.figure()
   ax = fig.add_subplot(1,1,1)
   plt.xlim(0,nsteps)
   plt.xlabel('steps')
   plt.ylabel('energy')
   # fci
   ehf  = -11.4467766233 
   efci = -11.5799784149 
   plt.axhline(y=ehf, linewidth=2, color='r')
   plt.axhline(y=efci, linewidth=2, color='b')
   plt.ylim(efci-0.01,ehf+0.02)
   plt.ion()
   plt.show()
   prefix = 'energy'
   # start
   saver = tf.train.Saver()
   init = tf.initialize_all_variables()
   with tf.Session() as sess:
      sess.run(init)
      try:
         saver.restore(sess, "./sites.ckpt")
      except:
         pass
      for i in range(nsteps):
	 valc = sess.run(optVal)
         print '\ni=',i,'valc=',valc,'norm=',\
	        sess.run(tree_dot(depth,sites,sites))
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
	 plt.pause(0.02)
	 if ifgif: plt.savefig(prefix+str(i)+".png",dpi=100)
         treelib.tree_save(sites,sess)
      # Save results
      save_path = saver.save(sess, "./sites.ckpt")
   if ifgif: anim.genGIF(prefix)
   return valc

#
# Binary trees
#
def tree_rand(depth,n,D1):
   isometries = []
   ni = n
   idx = 0
   for i in range(depth):
      nup = min(ni**2,D1)
      nlsite = 2**(depth-1-i)
      # top one
      if nlsite == 1:
         t = numpy.random.uniform(-1,1,ni*ni).reshape((1,ni,ni))
	 t = t/math.pow(ni*ni,0.25)
         isometries.append(tf.Variable(t))
         idx += 1
      else:
         for j in range(nlsite):
            t = numpy.random.uniform(-1,1,nup*ni*ni).reshape((nup,ni,ni))
	    t = t/math.pow(nup*ni*ni,0.25)
            isometries.append(tf.Variable(t))
            idx += 1
      ni = nup
   return isometries

def tree_rand0(depth,n,D1,occun,noise=1.e-2):
   print '\n[tree_rand0]'
   isometries = []
   ni = n
   idx = 0
   for i in range(depth):
      nup = min(ni**2,D1)
      nlsite = 2**(depth-1-i)
      # top one
      if nlsite == 1:
         t = numpy.random.uniform(-1,1,ni*ni).reshape((1,ni,ni))
         t = t*noise
	 t[0,0,0] = 1.0
	 isometries.append(t)
         idx += 1
      else:
         for j in range(nlsite):
            t = numpy.random.uniform(-1,1,nup*ni*ni).reshape((nup,ni,ni))
            t = t*noise
	    if i == 0:
	       icase1 = tflib.occIcase(occun,2*j)
	       icase2 = tflib.occIcase(occun,2*j+1)
	       t[0,icase1,icase2] = 1.0
	    else:
	       t[0,0,0] = 1.0
            isometries.append(t)
            idx += 1
      ni = nup
   # check 
   sites = [0]*len(isometries)
   for idx,site in enumerate(isometries):
      sites[idx] = tf.Variable(site)
   return sites

# <T1|T2>
def tree_dot(depth,t1,t2,debug=False):
   n1 = len(t1)
   n2 = len(t2)
   assert n1 == n2 
   nsites = 2**depth
   ovaSeed0 = [tf.constant(numpy.identity(4,dtype=numpy.float64))]*nsites
   idx = 0
   for i in range(depth):
      nlsite = 2**(depth-1-i)
      if debug: print '\n>>> i/nlsite=',i,nlsite
      ovaSeed1 = [0]*nlsite
      for j in range(nlsite):
	 s1 = ovaSeed0[2*j]
	 s2 = ovaSeed0[2*j+1]
	 bs = t1[idx] 
	 ks = t2[idx]
	 # contraction
	 #   |
	 #  / \
         # *   *
	 #  \ /
	 #   |
	 tmp = tflib.tensordot(bs,s1,axes=([1],[0]))  	  # (u,d1,d2),(d1,d1')=>(u,d2,d1')
	 tmp = tflib.tensordot(tmp,s2,axes=([1],[0])) 	  # (u,d2,d1'),(d2,d2')=>(u,d1',d2')
	 tmp = tflib.tensordot(tmp,ks,axes=([1,2],[1,2])) # (u,d1',d2'),(u',d1',d2')=>(u,u')
	 ovaSeed1[j] = tmp
         idx += 1
      ovaSeed0 = [ova for ova in ovaSeed1]
   ova = tf.reshape(ovaSeed0[0],[],name='ova')
   return ova

#
# Expectation value
#
def tree_cntr(depth,t1,t2,mpo):
   ovaSeed0 = [tf.constant(site) for site in mpo]
   idx = 0
   for i in range(depth):
      nlsite = 2**(depth-1-i)
      ovaSeed1 = [0]*nlsite
      for j in range(nlsite):
         s1 = ovaSeed0[2*j]
         s2 = ovaSeed0[2*j+1]
         bs = t1[idx] 
         ks = t2[idx]
         # contraction of mpo
         #     |
         #    / \
         #   |   |	 	    
         # --*---*--
	 #   |   |	 	    
   	 #    \ /
   	 #     |
	 # (u,d1,d2),(l,r,d1,d1')=>(u,d2,l,r,d1')
	 # (u,d2,l,r,d1'),(r,r',d2,d2')=>(u,l,d1',r',d2')
	 # (u,l,d1',r',d2'),(u',d1',d2')=>(u,l,r',u')
	 # (u,l,r',u')->(l,r',u,u')
	 tmp = tflib.tensordot(bs,s1,axes=([1],[2]))      
	 tmp = tflib.tensordot(tmp,s2,axes=([1,3],[2,0])) 
	 tmp = tflib.tensordot(tmp,ks,axes=([2,4],[1,2])) 
	 tmp = tf.transpose(tmp,perm=[1,2,0,3])   
	 ovaSeed1[j] = tmp
         idx += 1
      ovaSeed0 = [ova for ova in ovaSeed1]
   return tf.reshape(ovaSeed0[0],[])

# <T|H|T>
def calHexp(depth,t1,t2,hmpo,debug=True):
   if debug: print '\n[calHexp]'
   hpwts,mpos = hmpo
   nops = len(mpos)
   nsite = len(mpos[0])
   if debug: print ' nops/nsite=',nops,nsite
   n1 = len(t1)
   n2 = len(t2)
   assert n1 == n2 
   # sum over operators: ex = <Psi|Hx|Psi>
   Hpp = tf.Variable(0.,dtype=tf.float64)
   for iop in range(nops):
      # for each operator, we do the renormalization along tree depth
      contraction = tree_cntr(depth,t1,t2,mpos[iop])
      Hpp += tf.mul(contraction,hpwts[iop])
   return Hpp 

# N-mpo
def genNmpo(nsite):
   iden = numpy.identity(4)
   nmat = numpy.zeros((4,4))
   nmat[1,1] = 1.0
   nmat[2,2] = 1.0
   nmat[3,3] = 2.0
   nmpo = [0]*nsite
   # s0
   tmp = numpy.zeros((1,2,4,4))
   tmp[0,0] = iden
   tmp[0,1] = nmat
   nmpo[0] = tmp
   # s0-s[-1]
   tmp = numpy.zeros((2,2,4,4))
   tmp[0,0] = iden
   tmp[0,1] = nmat
   tmp[1,1] = iden
   for i in range(1,nsite-1):
      nmpo[i] = tmp
   # s[-1]
   tmp = numpy.zeros((2,1,4,4))
   tmp[0,0] = nmat
   tmp[1,0] = iden
   nmpo[-1] = tmp
   return nmpo

# <T|N|T>
def calNexp(depth,t1,debug=True):
   if debug: print '\n[calNexp]'
   nsite = 2**depth
   nmpo = genNmpo(nsite)
   Hpp = tree_cntr(depth,t1,t1,nmpo)
   return Hpp 

if __name__ == '__main__':
   
   import tflib
   L = 8
   n = 4
   # ValueError: Cannot create a tensor proto whose content is larger than 2GB.
   # D1 = 1 -> Relax to Hartree-Fock energy
   D1 = 10
   occun = [1]*L+[0]*L 

   depth = int(numpy.log2(L))

   print '\ntree with depth =',depth
   #sites = tree_rand(depth,n,D1)
   sites = tree_rand0(depth,n,D1,occun)
   nsite = len(sites)
   for isite in range(nsite):
      print isite,sites[isite].get_shape()

   normalization = tree_dot(depth,sites,sites)

   Npp = calNexp(depth,sites)
   hmpo = loadHmpo('hop8') 
   Hpp = calHexp(depth,sites,sites,hmpo)
   energy = tf.div(Hpp,normalization)

   #import h6
   #mps1 = tflib.mps_rand0(L,n,D1,occun,noise=0.)
   #mps1_normalization = tflib.mps_dot(mps1,mps1)
   #mps1_Hpp = h6.calHexp(mps1,'hop8')
   #mps1_energy = tf.div(mps1_Hpp,mps1_normalization)

   init = tf.initialize_all_variables()
   with tf.Session() as sess:
      sess.run(init)
      #print
      #print 'check_MPS'
      #print '<P|P>  =',sess.run(mps1_normalization)
      #print '<P|H|P>=',sess.run(mps1_Hpp)
      #print 'Energy =',sess.run(mps1_energy)
      print
      print 'check'
      print '<P|P>  =',sess.run(normalization)
      print '<P|N|P>=',sess.run(Npp)
      print '<P|H|P>=',sess.run(Hpp)
      print 'Energy =',sess.run(energy)

   optimize([depth,sites,hmpo],energy,nsteps=5)
