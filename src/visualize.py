import numpy as np
import imageio
import skimage.transform
import matplotlib.pyplot as plt

import os

np_dir = './numpy_array/'
saving_dir = './gif/'
input_filenames = ['gradient_images', 'std_images']
saving_filenames = ['gradient', 'std']

def set_config(_training_type='STDL', _gate_activation='elu', _output_activation='elu'):
   global training_type, gate_activation, output_activation, \
          _name, files, choose_num
   training_type = _training_type
   gate_activation = _gate_activation
   output_activation = _output_activation
   _name = training_type + '_Gate_Activation_' + gate_activation + '_Output_activation_' + output_activation
   files = [x for x in os.listdir(np_dir) if _name in x]
   files = [x for x in files if input_filenames[0] in x]
   files = sorted(files)
   choose_num = '9'
set_config()

def proc_a_img(img, normalize, delta, max_min_pair=None):
   if delta:
      img_tmp = np.concatenate([np.expand_dims(img[0], 0), img[:-1]], axis=0)
      img = img / img_tmp
   if normalize:
      mean = np.mean(img)
      std = np.std(img)
      img = (img - mean) / std
   _max = np.amax(img)
   _min = np.amin(img)
   if max_min_pair is not None:
      _max, _min = max_min_pair
   assert _max != _min
   img = (img - _min) / (_max - _min)
   return img
   

def proc_imgs(filename, save_name, max_min_pair=None, ref_file=None, normalize=True, delta=True):
   np_img = os.path.join(np_dir, filename)
   np_img = np.load(np_img)
   buf = []
   if ref_file is None:
      do_ref = [[]] * len(np_img.files)
   else:
      ref_img = os.path.join(np_dir, ref_file)
      ref_img = np.load(ref_img)
      do_ref = ref_img.files
   for img_symbol, ref_img_symbol in zip(np_img.files, do_ref):
      imgs = np_img[img_symbol]
      if ref_file is not None:
         img = imgs / ref_img[ref_img_symbol]
      imgs = np.array([proc_a_img(img, normalize, delta, max_min_pair) for img in imgs])
      print imgs.shape
      _save_name = save_name + img_symbol + '.gif'
      imageio.mimwrite(os.path.join(saving_dir, _save_name), imgs, fps=50)

def get_min_max():
   _max = -np.inf
   _min = np.inf
   for filename in files:
      np_img = os.path.join(np_dir, filename)
      np_img = np.load(np_img)
      for img_symbol in np_img.files:
         imgs = np_img[img_symbol]
      tmp = np.amax(imgs)
      if tmp >= _max:
         _max = tmp
      tmp = np.amin(imgs)
      if tmp <= _min:
         _min = tmp
   return _max, _min

def do_mean_img(filename):
   np_img = os.path.join(np_dir, filename)
   np_img = np.load(np_img)
   buf = []
   np_img = np.mean(np_img[choose_num], axis=-1)
   print np_img.shape
   # (timestep, )
   np.set_printoptions(threshold=np.nan)
   return np_img

def a_layer_analysis(filename):
   img = do_mean_img(filename)
   mean = np.mean(img, axis=-1)
   print filename, np.mean(mean)
   return mean
#   print filename, mean

def layer_analysis():
   buf = []
   all_mean_buf = []
   for idx, filename in enumerate(files):
      layer_img = a_layer_analysis(filename)
      buf.append(layer_img)
      all_mean_buf.append(np.mean(layer_img))
   all_mean_buf = list(reversed(all_mean_buf))
   plt.scatter(range(len(all_mean_buf)), all_mean_buf)
   plt.plot(range(len(all_mean_buf)), all_mean_buf, label=gate_activation)
   plt.axis([0, len(all_mean_buf), 1e-4, 2.5e-3])
   buf = np.array(buf)
   buf = proc_a_img(buf, False, False)
   s = buf.shape
   buf = skimage.transform.resize(buf, [s[0], 20])
   print buf.shape
#   imageio.imwrite(os.path.join(saving_dir, _name + 'layer_analysis.png'), buf)

def multiple_activation_compare():
   activation_candidate = ['elu', 'relu', 'leaky_relu', 'tanh']
   for act in activation_candidate:
      set_config('STDL', act, act)
      layer_analysis()
   plt.xlabel('Num_passed_layers')
   plt.ylabel('1norm_Gradient')
   plt.legend()
#   plt.show()
   plt.savefig('multiple_activation_compare.png')

multiple_activation_compare()
