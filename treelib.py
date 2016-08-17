import numpy
import h5py

def tree_save(sites,sess,fname='sites.h5'):
   f = h5py.File(fname,'w')
   nsite = len(sites)
   f['nsite'] = nsite 
   for isite in range(nsite):
      npsite = sess.run(sites[isite])
      f['site'+str(isite)] = npsite
   f.close()
   return 0

def tree_load(fname='sites.h5'):
   f = h5py.File(fname,'w')
   nsite = f['nsite'].value
   sites = [0]*nsite
   for isite in range(nsite):
      sites[isite] = f['site'+str(isite)].value
   f.close()
   return 0
