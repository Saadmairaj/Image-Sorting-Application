# Image Sorting Application

We can sort out different kinds of images in well organised manner using this application. It is designed with pyqt5 and uses TensorFlow API and COCO pre-trained models to detect objects in the images which are then re-oganised in different folders according to their object names. After selecting required images we select the type of pre-trained model like _(ssdlite, ssd, fast-rcnn, faster-rcnn)_ then start the processing. The program will start going through every single image and separate them one by one according to the objects found in them. It can also be configured to save detection of the objects found in the images.

## Features of the application

There are multiple options to configure for desired results:-

1. *Image previewer:-* This window will show the current working image.
2. *Image:-* We change the number of images required from the total images found
3. *Threshold:-* We can set the threshold for the detection.

4. *Checkboxes of the application:-*
   - *Show Visualization:-* To see the bounding boxes in the images.
   - *Save Visualization:-* To save those bounding boxes with the images.
   - *Go through folders:-* This will walk and fetches images from folders inside folders.
   - *All Objects:-* Every object detected by the model will also count.

5. *Search bar:-* It is to search all the images containing the searched object for example if we search for “Cat” it only sort all the images of cat to one folder and ignore any other object which is not a cat.

6. *Select model:-* We can select different type of pre-trained models from the given list.

7. *Select Directory:-* Path to the directory to fetch the images.

8. *Save Directory:-* Path to the directory where the save will be saved by default it will save to the selected directory path.

9. *Start:-* When all the necessary information is fed then start button will enable and by pressing it will start the process of sorting.

<p align='center'><img src="https://raw.githubusercontent.com/Saadmairaj/Image-Sorting-Application/master/sample.png"></p>
