<div align="center">
  <a href="https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation">
    <img src="Images/med_img_seg.png" alt="Logo" width="" height="200">
  </a>

<h1 align="center">Medica Image Segmentation</h1>
</div>


## 1. Problem Statement
Image segmentation is a vital aspect of image processing, differentiating itself from image classification by providing pixel-level identification of image content. It involves dividing an image into distinct regions based on features like color, texture, and shape, enabling internal consistency within regions and clear distinctions between them.
</br>
medical image segmentation is a critical prerequisite for healthcare systems, aiding in disease diagnosis and treatment planning. It analyze and process 2D or 3D images to achieve segmentation, extraction, three-dimensional reconstruction and three-dimensional display of human organs, soft tissues and diseased bodies. 
With the rapid development of deep learning, image segmentation methods, have achieved good results and it encompasses various applications, from the brain and eyes to the chest, abdomen, and heart [1].


## 2. Related Works
The medical image segmentation network uses an encoder-decoder structure. The encoder extracts features from the input image and turns them into a low-resolution map. The decoder then takes this map and labels each pixel in high detail to realize the category labeling of each pixel.
</br>
</br>
The first successful deep learning network for image segmentation was the fully convolutional network. It paved the way for using convolutional neural networks in this task. Then there are Other networks like U-Net, Mask R-CNN, RefineNet, and DeconvNet, which have a strong advantage in processing fine edges.
U-Net, which uses a U-shaped architecture with skip-connections, has become a standard for medical image segmentation, achieving great success. Transformers are also being used, as seen in papers like UTNet and TransUNet. They combine Transformers and Convolutional Neural Networks for better results  and use benefits of both networks.
</br>
</br>
We could classifies deep learning-based medical image segmentation into four categories: FCN, U-Net, GAN, and Transformers. See the figure below for these categories with examples for each [1].
</br>
</br>
<img src="Images/chart.png" width="">
</br>
</br>
<b>Loss functions:</b>
</br>
Loss functions that are used more in these models are:
Cross entropy loss, Weighted cross entropy loss, Dice loss, Tversky loss, Generalised dice loss, Boundary loss, Exponential logarithmic loss and ... [2].
</br>
</br>
<b>Evaluation metrics:</b>
</br>
The evaluation of image segmentation performance relies on pixel quality, region quality and surface distance quality. Some popular metrics are: 
Pixel quality metrics include pixel accuracy (PA). Region quality metrics include Dice score, volume overlap error (VOE) and relative volume difference (RVD). Surface distance quality metrics include average symmetric surface distance (ASD) and maximum symmetric surface distance (MSD) [2].


## 3. The Proposed Method
<b>Medical image segmentation involves several key steps:</b>

1. Obtain a dataset and divide it into training, testing, and validation sets.

2. Preprocess the images, which typically includes standardizing the input images and augmenting the dataset by applying random rotations and scalings to increase its size.

3. Apply a suitable medical image segmentation method.

4. Classification.

5. Performance metrics and validate the results.
</br>
Its block diagram is shown in the figure below:
</br>
</br>
<img src="Images/chart2.png">
</br>

## 4. Implementation


### 4.1. Dataset
Under this subsection, you'll find information about the dataset used for the medical image segmentation task. It includes details about the dataset source, size, composition, preprocessing, and loading applied to it.
[Dataset](https://drive.google.com/file/d/1-2ggesSU3agSBKpH-9siKyyCYfbo3Ixm/view?usp=sharing)

### 4.2. Model



### 4.3. Configurations


### 4.4. Train

### 4.5. Evaluate

## 5. Refrences
[1]: Liu, X., Song, L., Liu, S., & Zhang, Y. (2021). A review of deep-learning-based medical image segmentation methods. Sustainability, 13(3), 1224.
</br>
[2]: Wang, R., Lei, T., Cui, R., Zhang, B., Meng, H., & Nandi, A. K. (2022). Medical image segmentation using deep learning: A survey. IET Image Processing, 16(5), 1243-1267.
