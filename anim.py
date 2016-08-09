from images2gif import writeGif
from PIL import Image
import os

def genGIF(prefix,ifdel=True):
   #Stores a name-list of jpg and png files into the variable file_names.
   #Note: endwiths can be changes to load other image types
   flst = [fn for fn in os.listdir('.') \
	   if (fn.endswith('.png') or fn.endswith('.jpg')) and fn.startswith(prefix)]
   
   if len(flst) == 0:
      print 'no such files with prefix = ',prefix
      return 0

   file_names = sorted(flst)

   #Open and stores all files by name into the variable images, it also converts the images into RGB format
   images = [Image.open(fn).convert('RGB') for fn in file_names]
    
   #Size of the gif
   size = (400,400)
    
   #Makes the images smaller
   for im in images:
       im.thumbnail(size, Image.ANTIALIAS)
    
   #Name of the gif
   filename = prefix+".gif";
    
   #Converts the images into a gif with a desired duration between the images
   writeGif(filename, images, duration=0.1)
    
   print "Done with ",filename

   if ifdel:
      for fl in file_names:
         os.remove(fl)
   return 0



if __name__ == '__main__':

   genGIF('energy')
