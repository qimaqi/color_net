# Let there be Color!-pytorch
## Background
This project is adapted from [repo](https://github.com/shufanwu/colorNet-pytorch) and compatible with the leonhard environment
>cudnn/7.6.4
>cuda/10.1.243
>python_gpu/3.8.5

A Neural Network For Automatic Image Colorization was build in Pytorch and was used for colorize synthesized images from reconstruction network.

Check out more detail from original website [Let there be color](http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/en/). 



## Overview
* Net model
![...](https://github.com/AlexanderNevada/color_net/blob/master/readme%20images/model.png)

* Dataset: 

  Selected Megadepth and NYU was used for training, you can run code below to download:

  > wget --no-check-certificate "https://onedrive.live.com/download?cid=96513E40200F96F7&resid=96513E40200F96F7%21392&authkey=ADRMduh2_kMyys8" 

- Since our project do not need classification task so corresponding classification loss or dataset is removed.

## Training

- Train with dataset above or your own data

  >Hint: Put all images in one subfolder and keep it consistent with 'data_dir' in train.py
  >
  >For example:
  >
  >----- data_dir
  >
  >​	   ----- train
  >
  >​				----- image01.jpg
  >
  >​				----- image02.jpg
  >
  >​				----- image03.jpg
  >
  >​				...
  >
  >---

- Then hyperparameter should be change for your own use in train.py

  > Hint: Epochs, data_dir, dir_checkpoint and save_name

- If you want to test the model through val.py you have to first transfer the color image to gray

  > Hint: use function like below will solve this question

  ```python
  from PIL import Image
  def read_image(impath,resize_scale):
      color_img = Image.open(impath)
      w,h = color_img.size
      new_w = int(resize_scale*w)
      new_h = int(resize_scale*h)
      resize_color_img = color_img.resize((new_w, new_h), Image.ANTIALIAS)
      resize_gray_img = resize_color_img.convert('L')
      # return crop_img, crop_gray
      return resize_color_img, resize_gray_img
  ```



## Result

64 Epochs are trained and result is shown below:

- process results

  ![...](https://github.com/AlexanderNevada/color_net/blob/master/readme%20images/color_change_result.gif)  

* Good results  
![...](https://github.com/AlexanderNevada/color_net/blob/master/readme%20images/good-result.png)  
* Bad results  
![...](https://github.com/AlexanderNevada/color_net/blob/master/readme%20images/bad_result_1.jpg)  
* ![...](https://github.com/AlexanderNevada/color_net/blob/master/readme%20images/bad_result_2.jpg)
* Indoor performs not as good as outdoor since unbalanced dataset

## Pretrained model
You can download the model: colorization.pkl from https://drive.google.com/drive/folders/17WY-RxN3G3uLBclI_wvftXMQZWIwd6q8?usp=sharing



