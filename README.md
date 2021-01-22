# instance_segmentation
Instance Segmentation with combined model of Yolo v3 + FCN (Pytorch)

yolov3.cfg (236 MB COCO Yolo v3) - requires 4 GB GPU-RAM: https://pjreddie.com/media/files/yolov3.weights
fcn8_pascal_500.pth (512MB fcn weight, you can also train by yourself): https://drive.google.com/file/d/1AIlxg-H6KQkR8FIxMDeFoTOKM8_Z7p_s/view?usp=sharing

## Introduction:
For many computer vision problems, simply classifying the class an object belongs to is not enough. A lot of times identifying the shape of an object and the number of times the object appears in the image is required. Many algorithms go about solving this a little differently however one thing is usually common is that where one would usually find a full-connected layer, it is replaced by some deconvolutional / upsampling layers. As a result, the output maintains the original size of the image input whilst also highlighting which pixels in the image correspond to which class. One of the state-of-the-art models within this research area is Mask-R-CNN. The model expands upon the work of Faster-R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. The backbone used for the model was one of the Resnet architectures to extract features from the image to which it is then passed through a Region Proposal Network (RPN) to generate Regions Of Interest (ROI) from the feature map. A ROI-Align network is then applied warp the ROI’s into a fixed dimension. This is then fed into fully connected layers to make an object classification and bounding box prediction and into the mask classifier to generate the image mask, all in parallel. 

## Yolo v3 + FCN
The model proposed here consists of 4 main stages. In the first stage, the images are input to the Yolo Version 3 network which uses Darknet-53 as its backbone. The model outputs the object classes, confidences of the detection and bounding boxes of each object. The Yolo network was pretrained on the COCO dataset. The output bounding boxes then were extracted as separate images in stage 2. All of these bounding boxes images were input to the pretrained Fully Convolutional Network (FCN) for pixel-wise classification, to generate the mask for the object in bounding box. In the final stage, the masks were reproduced in each separate bounding box image. Finally, the masks were pasted back to the original bounding box location. 

![](https://github.com/namm2008/instance_segmentation/blob/main/example/new%20model%20structure.png)

*Fig.1 New Model Structure*

## Dataset
The dataset used was the subset of COCO dataset. COCO dataset is renowned for its well labeled data and it was used as benchmarks for many image classifications. In our investigation, we sorted out 5 categories, ‘person’, ‘chair’, ‘couch’, ‘dining table’ and ‘toilet’. The ground truth masks only include those 5 groups. We had 2 main reasons to choose these 5 categories. Those 5 classes of object appear to be observed indoor. As per our research domain, the indoor environment should be more relevant. The second reason being the time concern. Focusing on 5 categories would be enough to test the model.

To build our own subset of COCO, a new json file of the same format as the COCO annotations files was created, and by using the COCO API, we parsed through the entire dataset extracting all the image and annotation information pertaining to our 5 classes and appending each into a list. As a result of this we could easily extract all the images necessary into a separate file. In addition to this since our json file was of the COCO format, we could use the COCO API to extract the masks and bounding boxes when building our dataset class for our data-loader.


## Implementation:
### Stage 1: Deploying Yolo
In this stage, we use of Yolo v3 to detect the images and outputting the bounding boxes, confidence scores and classes number. The backbone network used in Yolo v3 is the Darknet-53. In our implementation, the Darknet was provided by source code. The network was pretrained on the COCO dataset where the pretrained weight parameters was from the original paper. 

To begin with, the inputting images have to be preprocessed to resize into (Batch x Channel x Height x Width). The height and width size should be equal to (256x256) in order to match the neural network. After a single forward pass through Darknet, the output tensor is a size of (Batch size x Observation x 85). Observation number is 4032. The Darknet-53 output observation in 3 different scales (192, 768 and 3072 Observations) which give a total number of 4032. The final dimension of the output was 85 which consists of the components, bounding box coordinates (4), confidence scores (1), and classes confidences (80). The output tensor carried out non-max suppression which is outputting the bounding boxes with highest class confidences scores. After this, the observation with confidence scores lower than a threshold value would also be suppressed. To be noted, this threshold value is also a hyperparameters to be tuned and revised. Initially, the threshold value was set to be 0.75.

As mentioned above, the model was loaded with the pretrained weights on COCO dataset. One thing we needed to handle was to suppress the classes that were not interested in. After that, the output from the above process was then converted to a prediction variable which has a shape of (Number of Observation remaining x 8). The 8 columns included the Image Number (1), Bounding Box Coordinates (4), Class Confidence (1), Scores (1), and Class Number (1). The bounding box coordinates was converted in order to match the original shape of image. These tensors included the useful information for the next stage process. 

### Stage 2: Extraction Bounding Box
In this stage, the prediction variable was sliced to include only the bounding box coordinates and changed back to integer values in order to get the correct coordinates. The bounding box image was an array due to the slicing from the original image. Transforming to tensor was carried out which fulfil the Pytorch setting requirement. 
![](https://github.com/namm2008/instance_segmentation/blob/main/example/Extraction%20BB.png)

*Fig.2 Exrration of Bounding Box*

### Stage 3: Training and Deploy FCN8s
As discussed before FCN-8 had been already trained on the Pascal-VOC 2012 dataset before to produce very good results. However, in our case we are interested in using FCN-8 as a binary classifier (only generating a mask for the background and object). For this to work the last six convolutional were modified to make predictions for two classes only. As results new weights are generated using the He method of initialization. Furthermore, the Pascal-VOC 2012 data used to train the model was slightly modified to work for binary classification. Since we are no longer interested in the model’s ability in distinguishing between different classes in the image, the ground truths are modified to produce a single mask for all the classes in the image. An example of this can be seen in Fig. 3, whereby a single mask is produced for the two classes present. All images into model were resized to (224,224) to allow for the use of mini batches. Also, the pre-processing of the images followed that of the research behind FCN-8 in which the per channel mean is subtracted from the image tensor.

![](https://github.com/namm2008/instance_segmentation/blob/main/example/modification%20to%20GT.png)

*Fig.3 Modification of GT*

Before finetuning, the gradient on all the layers but the newly initialized layer was frozen resulting in the number of trainable parameters to be 10,886. The Cross-entropy loss function was used to evaluate the masks and Adaptive Moment Estimation (Adam) optimizer was used in the backpropagation. The learning rate was initially set to 0.001 and was set to decay a tiny bit after every epoch. The metric used to evaluate the quality of the masks produced is the Mean IoU. These were chosen since the researchers of original FCN-8 model used the same metrics to evaluate results on the Pascal-VOC 2012 dataset. 

The model was finetuned for a total 400 epochs resulting with a total training time of roughly 12 hours. The plots of training and validation loss can be seen Fig. 3.1 below. The model looks to have converged after only 200 epochs. In addition to this some overfitting is present due to the distance between training and validation plots. Furthermore, the IoU for each image in the training set is calculated and the distribution of IoU’s can be seen in Figure 3.31. It can be seen that the majority images in test set resulted in an IoU greater than 0.5. The Mean IoU for the test set was calculated to be 0.614 which is impressive considering that in the original FCN-8 model achieved a mean IoU of 0.622. In addition, the finetuned model had an inference time 455ms on a GPU, a little slower than the original model. 

![](https://github.com/namm2008/instance_segmentation/blob/main/example/Loss.png)

*Fig 3.1 Training and Validation loss (left) distribution of IoUs (right)*


### Stage 4: Masking Process
In this stage, we aimed at combining all the information obtained from the above to form the instance segmentation. FCN8s outputted a 2 layers tensor with shape equal to the inputting dimension. In here, we detached the tensors and change to Numpy arrays for easier manipulation. The masks were padded with Zeros for the areas not masked to the original shape. 

In the last procedure, we need to handle the mask area overlapping problem. Objects in two overlapped bounding boxes showed in the Fig 3.4, the green bounding box and the red bounding box also cover the person. There was overlapping area between the two bounding boxes. Both the green box and the red box mask the head of the person. As we only separate the background and the object by binary classifier, the green box may treat the head of person as object and mask it. 

To solve this problem, the confidence scores of the objects in both bounding boxes would be compared. In the situation overlapping, one with higher confidence scores was used to mask the object. The overlapping area was eliminated from the one with smaller confidence scores. For example, in Fig 3.4, the confidence scores of red bounding boxes was 0.9678 and the green one was 0.8998. As the score of the red box was higher than the green one, the overlapping area belonged to the red box. 

![](https://github.com/namm2008/instance_segmentation/blob/main/example/overlapping%20bb.png)

*Fig. 4 Overlapping handling*

## Example Result:
There are some of example output showing the new model(Left) and the state of the art Mask RCNN model(Right)
![](https://github.com/namm2008/instance_segmentation/blob/main/example/ex1.png)
![](https://github.com/namm2008/instance_segmentation/blob/main/example/ex2.png)
![](https://github.com/namm2008/instance_segmentation/blob/main/example/ex3.png)
![](https://github.com/namm2008/instance_segmentation/blob/main/example/ex4.png)
![](https://github.com/namm2008/instance_segmentation/blob/main/example/ex5.png)
![](https://github.com/namm2008/instance_segmentation/blob/main/example/ex6.png)
![](https://github.com/namm2008/instance_segmentation/blob/main/example/ex7.png)

## Actknowledgement
The coding of Yolo v3 is from github repo Detectx-Yolo-V3 (https://github.com/AyushExel/Detectx-Yolo-V3)
