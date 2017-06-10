import numpy as np
import imageio
import os
np_dir = '../numpy_array/'
saving_dir = '../gif/'
input_filenames = ['gradient_images', 'std_images']
saving_filenames = ['gradient', 'std']
grad_imgs = np.load(os.path.join(np_dir, input_filenames[0]))
std_imgs = np.load(os.path.join(np_dir, input_filenames[1]))

def proc_a_img(img, normalize, delta):
   if delta:
      img_tmp = np.concatenate([np.expand_dims(img[0], 0), img[:-1]], axis=0)
      img = img / img_tmp
   if normalize:
      mean = np.mean(img)
      std = np.std(img)
      img = (img - mean) / std
      print mean, std
   _max = np.amax(img)
   _min = np.amin(img)
   img = (img - _min) / (_max - _min)
   return img
   

def proc_imgs(np_img, save_name, normalize=True, delta=True):
   buf = []
   for img_symbol in np_img.files:
      imgs = np_img[img_symbol]
      imgs = np.array([proc_a_img(img, normalize, delta) for img in imgs])
      print imgs.shape
      _save_name = save_name + img_symbol + '.gif'
      imageio.mimwrite(os.path.join(saving_dir, _save_name), imgs, fps=100)

proc_imgs(grad_imgs, saving_filenames[0], False)
proc_imgs(std_imgs, saving_filenames[1], False)
