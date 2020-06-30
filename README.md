# MT_CrossingAnalysis

This repo propose a pre-trained model *DGF-Net* for microtubule segmentation.


## Architecture of the DGF-Net
![](https://github.com/cbmi-group/MT_CrossingAnalysis/blob/master/dgfnet.png)



## How to use

### Dependencies
The main dependencies are as follows:

* Pytorch
* Python 3
* CV2

### Model
You can download the pre-trained model from [here](https://drive.google.com/file/d/1c-MILdzsqpagTFJNoYuS_ogVnxHDSpak/view?usp=sharing).

See model.py for details


### Data
Please put your source images into folder *images* (default), and the final segmentation images will be save in folder *segmentations* (default). 

Note: the size of the input images should not larger than 1024*1024. 

### Quick start 
run inference.py
```
python inference.py -t 0.6 --save_dir './segmentation' --img_dir './images' --img_type 16
```
You can use `-t` to change the threshold for different segmentation results,  use `--save_dir` to create the saving folder, use `--img_dir` to change your source image dir, use `--img_type` to change the type of images, default *uint16*. 

(Note that `--img_type` must match the soure image's type)


## Contributing 
Code for this projects developped at CBMI Group (Computational Biology and Machine Intelligence Group).

CBMI at National Laboratory of Pattern Recognition, INSTITUTE OF AUTOMATION, CHINESE ACADEMY OF SCIENCES

Bug reports and pull requests are welcome on GitHub at https://github.com/cbmi-group/MT_CrossingAnalysis

 
