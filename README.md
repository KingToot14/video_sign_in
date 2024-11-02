# Installation

## Install/Compile dlib
The `facial-recognition` packages requires dlib to be installed. For this, you can install and compile it from the [dlib website](http://dlib.net/compile.html).<br><br>
**Windows**<br>
Instead of manually compiling dlib, you can download the `dlib-19.24.99-cp312-cp312-win_amd64.whl` wheel file from [this github repo](https://github.com/z-mahmud22/Dlib_Windows_Python3.x).<br><br>
Then place the file in your project root and run<br>
```
pip install dlib-19.24.99-cp312-cp312-win_amd64.whl
```

## Installing Dependencies
Run the requirements.txt file using `pip install -r requirements.txt` to download all the dependencies

## Clean Up
You may now delete `requirements.txt` and the wheel file (if used), if you wish.

# Usage
You can train the model and start the video feed by using
```
python video_sign_in.py <args>
```
Depending on the number of known images, it may take some time to train the model

## Command Line Arguments
This program provides a few command line arguments in order to configure the model and video capture system

### **Debuging Mode** | -d , --debug
Provides extra debug information:
- Prints out the training time of each image/whole model
- Logs how long it takes to train the full model

**Default:** false

### **Repeat Sign-in Cooldown** | -c --cooldown
Determines how long to wait until logging the same person again (in seconds)

**Default:** 300 (5 minutes)

### **Wait Frames** | -f --frames
How many frames to wait until we detect and recognize faces (<= 1 means every frame)

**Default:** 1 (every frame)

### **Recognized Faces Directory** | -r --recdir
The directory where each recognized face image is stored

**Default:** "recognized_faces"

# Directory Structure
Each individual person is stored in its own subdirectory under the specified 'recognized faces directory'

Each subdirectory contains 1 or more images that correspond to that person

All other files/directories are ignored