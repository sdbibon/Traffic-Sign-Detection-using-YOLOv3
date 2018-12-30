# Traffic-Sign-Detection-using-YOLOv3

## DOWNLOAD INSTRUCTIONS
1. Go the the following link https://drive.google.com/file/d/1m9Hm0p_YrCrsWm7vZmZTbqVI1d9zu26_/view?usp=sharing and download the file.
2. Extract the file to any directory.
3. It will look like this.
![capture1](https://user-images.githubusercontent.com/16569879/50551012-c3400c80-0c37-11e9-84b1-f29b66249202.JPG)

4. Follow the instruction given below and for seeing the **test video results** you will find the folder YOLOv3_DemoVideos.


## TEST VIDEOS

1. Place the test videos in the Videos folder of Darknet_TrafficSign_Yolov3/build/darknet/x64
2. The test videos file should be named as TestVideo_1.mp4 and TestVideo2.mp4 , which would be input to the 2 executable batch files
3. If you want to change the name or file format of input videos or the location of the files, please update the batch files yolov3_batch.bat and yolov3_batch_2.bat files accordingly.

## EXECUTING FOR INPUT VIDEO

1. Execute the batch file yolov3_batch.bat in Darknet_TrafficSign_Yolov3/build folder to run traffic sign detection demo for TestVideo_1.mp4 and yolov3_batch_2.bat file for TestVideo_2.mp4
2. Update this batch file if you want to change the name of configuration file, input test video files, weights file or training data file
3. Press the escape button to stop executing the model for the test video

## EXECUTING FOR INPUT THROUGH WEBCAM

Execute the below command from the folder where darknet.exe is located, that will be Darknet_TrafficSign_Yolov3/build/darknet/x64, this will execute the darknet model for input through webcam of the executing device using the loaded weights, configuration file and as per the details mentioned in trainer.data file:
darknet.exe detector demo custom/trainer.data custom/yolov3.cfg weights/yolov3_14400.weights

## STEPS AND COMMAND FOR TRAINING CUSTOM DATASET

1. Place the image files with annotated labels (txt format) in the Darknet_TrafficSign_Yolov3/build/darknet/x64/data/obj folder.
2. Place the create_train_text.py python script available in Darknet_TrafficSign_Yolov3 folder, inside the folder where images and labels are kept. Execute this python script which will generate train.txt and test.txt files containing training and testing image file names respectively. Place the generated train.txt and test.txt files inside the Darknet_TrafficSign_Yolov3/build/darknet/x64/custom folder
3. The trainer.data file available in Darknet_TrafficSign_Yolov3/build/darknet/x64/custom folder which is used for training the model has information about training and testing (cross validation) image files to be used (mentioned in train.txt and test.txt), the number of classed to detect, the class names to identify and path to keep the generated weights files (1 weight file generated for every 100 steps)
4. Execute the below command from the folder where darknet.exe is located, that will be Darknet_TrafficSign_Yolov3/build/darknet/x64, this will train the darknet model using the available darknet53 convolutional network and as per the details mentioned in trainer.data file:
darknet.exe detector train custom/trainer.data custom/yolov3.cfg darknet53.conv.74
5. Modify the file trainer.data as per the needs, to change path of generated weights files, training/testing txt file names and/or path, number of classes/class names

## COMMAND TO GET THE EVALUATION PARAMETERS OF THE MODEL

darknet.exe detector map custom/trainer.data custom/yolov3.cfg weights/yolov3_14400.weights

## MODIFYING THE CONFIG FILE FOR CUSTOM DATASET

• Line 3: set batch=24, this means we will be using 24 images for every training step
• Line 4: set subdivisions=8, the batch will be divided by 8 to decrease GPU VRAM requirements.
• Line 127: set filters=(classes + 5)*3 in our case filters=72 as number of classes=19
• Line 135: set classes=2, the number of categories we want to detect
• Line 171: set filters=(classes + 5)*3 in our case filters=72
• Line 177: set classes=19, the number of categories we want to detect

## SAMPLE FORMAT OF ANNOTATED TEXT FILES FOR CUSTOM IMAGES TO TRAIN

**1 0.716797 0.395833 0.216406 0.147222 0 0.687109 0.379167 0.255469 0.158333 1 0.420312 0.395833 0.140625 0.166667**
Where 1st column is the class_id , next 4 columns are the bounding box co-ordinates.
You can use tool like labelimg to draw bounding boxes on input images and generate annotated text files for yolo v3.

## INSTRUCTION FOR COMPILATION OF DARKNET (IN WINDOWS)

1. If you have MSVS 2015, CUDA 10.0, cuDNN 7.4 and OpenCV 3.x (with paths: C:\opencv_3.0\opencv\build\include & C:\opencv_3.0\opencv\build\x64\vc14\lib), then
3
start MSVS, open build\darknet\darknet.sln, set x64 and Releasehttps://hsto.org/webt/uh/fk/-e/uhfk-eb0q-hwd9hsxhrikbokd6u.jpeg and do the: Build -> Build darknet.
2. Also add Windows system variable cudnn with path to CUDNN: https://hsto.org/files/a49/3dc/fc4/a493dcfc4bd34a1295fd15e0e2e01f26.jpg NOTE: If installing OpenCV, use OpenCV 3.4.0 or earlier. 1.1. Find files opencv_world320.dll and opencv_ffmpeg320_64.dll (or opencv_world340.dll and opencv_ffmpeg340_64.dll) in C:\opencv_3.0\opencv\build\x64\vc14\bin and put it near with darknet.exe
3. Check that there are bin and include folders in the C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1 if aren't, then copy them to this folder from the path where is CUDA installed
4. To install CUDNN (speedup neural network), do the following:
o download and install cuDNN v7.4.1 for CUDA 10.0: https://developer.nvidia.com/cudnn
o add Windows system variable cudnn with path to CUDNN: https://hsto.org/files/a49/3dc/fc4/a493dcfc4bd34a1295fd15e0e2e01f26.jpg
5. If you want to build without CUDNN then: open \darknet.sln -> (right click on project) -> properties -> C/C++ -> Preprocessor -> Preprocessor Definitions, and remove this: CUDNN;
6. If you have other version of CUDA (not 10.0) then open build\darknet\darknet.vcxproj by using Notepad, find 2 places with "CUDA 10.0" and change it to your CUDA-version, then do step 1
7. If you don't have GPU, but have MSVS 2015 and OpenCV 3.0 (with paths: C:\opencv_3.0\opencv\build\include & C:\opencv_3.0\opencv\build\x64\vc14\lib), then start MSVS, open build\darknet\darknet_no_gpu.sln, set x64 and Release, and do the: Build -> Build darknet_no_gpu
8. If you have OpenCV 2.4.13 instead of 3.0 then you should change path after \darknet.sln is opened (right click on project) -> properties -> C/C++ -> General -> Additional Include Directories:C:\opencv_2.4.13\opencv\build\include (right click on project) -> properties -> Linker -> General -> Additional Library Directories: C:\opencv_2.4.13\opencv\build\x64\vc14\lib
9. If you have GPU with Tensor Cores (nVidia Titan V / Tesla V100 / DGX-2 and later) speedup Detection 3x, Training 2x:\darknet.sln -> (right click on project) -> properties -> C/C++ -> Preprocessor -> Preprocessor Definitions, and add here: CUDNN_HALF;
Note: CUDA must be installed only after that MSVS2015 had been installed.
Note: If you further compilation issue using Visual Studio 2015 when trying to build darknet, please check the opencv version (it has to be 3.4.0.12 or lower). Also, in some of the darknet files, opencv header files are referred, change the path in those source files which is giving the compilation issue to the path where open cv is installed in the current system.


