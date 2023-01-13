
# AdapNet:  Adaptive Semantic  Segmentation in  Adverse Environmental Conditions
AdapNet is a deep learning model for semantic image segmentation, aiming at assigning semantic labels (e.g., car, road, tree, etc.) to every pixel in the input image. Workable on a single GPU with 12 GB memory, AdapNet boasts a swift inference time. It sees application in benchmarks like Cityscapes, Synthia, ScanNet, SUN RGB-D, and the Freiburg Forest datasets.

DreamLens offers the TensorFlow implementation of AdapNet allowing training of personal models on any dataset and evaluation of the results through the mean IoU metric.

Adapnet can also be used with the [CMoDE](https://github.com/DreamLens/CMoDE) fusion scheme for multimodal semantic segmentation.

For citing our work, please find the reference:
```
@inproceedings{valada2017icra,
  author = {AdapNet project team},
  title = {AdapNet: Adaptive Semantic Segmentation in Adverse Environmental Conditions},
  booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  pages={4644--4651},
  year = {2017},
  organization={IEEE}
}
```

## Live Demo
http://deepscene.cs.uni-freiburg.de

## Segmentation Results
| Dataset       | RGB_Image     | Segmented_Image|
| ------------- | ------------- | -------------  |
| Cityscapes    |<img src="images/city.png" width=300> |  <img src="images/city_v1.png" width=300>| 
| Forest  | <img src="images/forest.png" width=300>  |<img src="images/forest_v1.png" width=300> |
| Sun RGB-D  | <img src="images/sun.png" width=300>  | <img src="images/sun_p.png" width=300>| 
| Synthia  | <img src="images/synthia.png" width=300>  | <img src="images/synthia_v1.png" width=300> |
| ScanNet v2  | <img src="images/scannet.png" width=300>  |<img src="images/scannet_pr.png" width=300> |

## System Requirements
#### Programming Language
```
Python 2.7
```
#### Python Packages
```
tensorflow-gpu 1.4.0
```
## Configure the Network
Download the resnet_v1_50 tensorflow pre-trained model for network intialization from [here](https://github.com/tensorflow/models/tree/master/research/slim).
#### Data
* Augment the training data. Resize the images in the dataset to 768x384 pixels and then apply random_flip, random_scale and random_crop.
* For each of the train, test, and val sets, run convert_to_tfrecords.py from dataset folder to create the tfrecords and the mean '.npy' file. Mean file should be created only for the train set.

Additional details are provided in the repository.

## Training and Evaluation
Please refer to the steps and guidelines given in the repository for the training and evaluation procedure.

## Additional Notes:
   * We provide the single scale evaluation script here.
   * This repo only performs training on a single GPU.

## License
For academic usage, the code is released under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license. For any commercial purpose, please contact DreamLens.