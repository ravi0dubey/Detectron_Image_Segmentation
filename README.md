# Detectron_Image_Segmentation 

## Problem Statement

We need a solution where in model is not only able to detect the face whether its of human(s) or animal(s), it should also be able to identify their names.
If we have 4 person and 2 animals (Cat,dog) whose faces gets clicked by the model and is stored in the system with the names of each person and the animals then once training happens model should be able to identify the person and the animals
as soon as their faces appear on the camera.


## Solution Proposed

In this project, the focus is to correctly detect the face and identify the face of the users/animals using deepinsight/InsightFace.
InsightFace is an integrated Python library for 2D&3D face analysis. It efficiently implements a rich variety of state of the art algorithms of face recognition, face detection and face alignment, which optimized for both training and deployment. </br>
**github link of InsightFace** https://github.com/deepinsight/insightface

## Tech Stack Used
1. Python </br>
2. MTCNN(Multi-task Cascaded Convolutional Networks)  https://pypi.org/project/mtcnn/
3. Keras to train the model </br>
  


# How to run the project
**Step 1** : We need to annotate the images using labelme and convert the annotation in cocoformat used by detectron.
              
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

      ![image](https://github.com/ravi0dubey/Detectron_Image_Segmentation/assets/38419795/11f1831e-ed98-4074-8103-79dd617f501e)                             

**Step 2** : Create a new environment
                command : conda create -n facerecognition python==3.6.9 -y </br>
    ![image](https://github.com/ravi0dubey/Detectron_Image_Segmentation/assets/38419795/11f1831e-ed98-4074-8103-79dd617f501e)            
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
