# Detectron_Image_Segmentation 

## Problem Statement

We need a solution where in model is able to detect the object be it person, animal or things(Car,bus etc) using the segmentation technique.

## What is Image Segmentation and why its better compared to bounding box object detection
Image segmentation serves as a foundation for various applications such as object recognition, medical imaging, autonomous vehicles, and more. It aims to partition an input image into meaningful and homogeneous regions, enabling computers to understand and analyze the visual content within the image.

a. It provides pixel-level accuracy, which allows for more precise object localization
b. When we need to count and distinguish individual instances of the same object class within an image like identifying individual cells in a microscope image often requires segmentation.
c. In case of irregular shapes, using bounding boxes can lead to a lot of empty space around the object. Segmentation, on the other hand, accurately captures the object's shape, reducing wasted space and improving efficiency.

## Challenges of Image Segmentation compared to Bounding box object detection

a. It is computationally Intensive especially when working with large images or real-time video streams. Bounding boxes are often more efficient.
b  It is complex to develop and train the segmentation models, especially deep learning-based ones, can be complex and requires large datasets.
c. It has some problem with accuracy as is is sensitive to image quality, noise, and variations in lighting and perspective. Bounding boxes can be more robust in some cases.
d. Annotation Effort is typically more time-consuming and labor-intensive compared to drawing bounding boxes.

## Solution Proposed
Keeping pros of segmentation into consideration, in this project, the focus is to correctly detect the object using detectron2 (https://github.com/facebookresearch/detectron2) </br>

## Tech Stack Used
1. Python </br>
2. model used is COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
3. Torch</br>
  


# How to run the project
**Step 1** : We need to annotate the images using labelme in polygon format and convert the annotation in cocoformat used by detectron.
              
            a. Install labelme by going to github link https://github.com/wkentaro/labelme and check the process of installation
            b. using command prompt or gitbash enter command "conda create -n segment python=3.7 -y"
            c. Then enter command "conda activate segment"
            d. the do the installation using command "pip install labelme"
            e. execute command "labelme" to run labelme
            f. select the directory where all images are stored. In my case the directory name is label_mask
            g. annotate the immage by creating polygon dot around the object. Below snapshot is of dog, so i created dots around the boundary and save it.
            h. Image and the annotation file in .json format will be saved in the label_mask folder. 
            i. now run "python labelme2coco.py label_mask" to convert the json format of annotation to coco format. Output coco format file trainval.json will get saved.
            j. We need to create one folder called "data" and save trainval.json file in it and create one subfolder "images" which will save all images.



   This is how annotation looks </br>
![image](https://github.com/ravi0dubey/Detectron_Image_Segmentation/assets/38419795/11f1831e-ed98-4074-8103-79dd617f501e)            
                        
**Step 2** : Create a new environment
                command : conda create -n facerecognition python==3.6.9 -y </br>
**Step 3** : activate your environment  </br>

                conda activate facerecognition  </br>
**Step 4** : conda install -c anaconda mxnet </br>

**Step 5** : conda install -c conda-forge dlib </br>

**Step 6** : Uninstall existing version of numpy and install numpy 1.16.1 version: </br>
        pip uninstall numpy </br>
        pip uninstall numpy </br>
        pip install numpy==1.16.1 </br>

**Step 7**:  Install requirements.txt in the newly created environment</br>
         pip install -r requirements.txt</br>

**Step 8** : Installation and setup is done:</br>
         a).  cd src</br>
         b). python app.py</br>


## Video link of project demo
https://youtu.be/MKOaQu3aXSs

## How project was designed and build
1. **app.py->** Driver program of the project which invokes the camera and then call subsquent method from each modules to perform the operations of collecting pictures from camera,training it and prediction of the face . </br>
2. **get_faces_from_camera.py->** Purpose is the get the 50 images from live feed of camera and crop the facial feature of the image and save it in 112 * 112 dimension </br> 
3. **faces_embedding.py->** Purpose of this class is to convert image into numerical value and saving it in pickel format. This process is called Face Embedding </br>
4. **train_softmax.py->** Purpose is to train the model using embeddings of the image. Model is trained in batchsize of 8 with 5 epochs. Relu activation for hidden layer and softmax for output layer. Saving the output as pickle format.</br>
5. **facePredictor.py->** Purpose is to do the prediction of the face. </br>
