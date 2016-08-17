import numpy
import tensorflow as tf

def skew(mat):
   amat = mat - tf.transpose(mat,perm=[1,0])
   return amat

# Q = (I-A)(I+A)^(-1)
def cayley(amat):
   shape = amat.get_shape()
   n = shape[0].value
   iden = tf.constant(numpy.identity(n))
   r = tf.matrix_inverse(tf.add(iden,amat))
   l = tf.sub(iden,amat)
   q = tf.matmul(l,r)
   return q

# Take Variable Matrix to Q
def utensor(dim,array1d,ifT=False):
   var = tf.reshape(array1d,[dim*dim,dim*dim])
   mat = cayley(skew(var))
   if ifT: mat = tf.transpose(mat,perm=[1,0])
   mat = tf.reshape(mat,[dim,dim,dim,dim])
   return mat

if __name__ == '__main__':

   n = 100 
   mat = numpy.random.uniform(-1,1,n*n).reshape((n,n))
   amat = mat-mat.T
   iden = numpy.identity(n)
   qmat = (iden-amat).dot(numpy.linalg.inv(iden+amat))
   print numpy.linalg.norm(qmat.dot(qmat.T)-iden)
   print 'qmat=',qmat

   tmat = tf.constant(mat)
   q = cayley(skew(tmat))
   idn = tf.constant(numpy.identity(n,dtype=numpy.float64))
   diff = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(tf.matmul(q,q,transpose_b=True),idn))))

   with tf.Session() as sess:
      print 'qtmat=',sess.run(q)
      print 'diff=',sess.run(diff)
