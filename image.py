import sys
from sklearn.decomposition import PCA
from PIL import Image
import scipy.io as sio
import numpy as np
from scipy.misc import imsave
image_fullpath=sys.argv[1]
image_name=sys.argv[2]
data=sio.loadmat(str(image_fullpath))
b=sio.whosmat(str(image_fullpath))
a=b[0]            
j=a[0]           
ndata=data[j] 
g=ndata.transpose(2,0,1).reshape(-1,ndata.shape[1])
pca = PCA(n_components=2)
g_new= pca.fit_transform(g)
sio.savemat('media/dataset/g_new.mat', {'g_new':g_new})
abc=np.dstack([ndata[:,:,10],ndata[:,:,0],ndata[:,:,60]])
#ab=ndata[:,:,12]
imsave('/home/venky/llc/media/image.png',abc)
im = np.array(Image.open('/home/venky/llc/media/image.png').convert('LA')) #you can pass multiple arguments in single line
#print(type(im))

gr_im= Image.fromarray(im).save('/home/venky/llc/media/temp.png')

#imsave('/home/venky/Desktop/llc/media/temp.png',ab)
