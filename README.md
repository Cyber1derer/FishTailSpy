# FishTailSpy

## CodecInfo.py
This Python script utilizes the OpenCV library to open a video file and retrieve its essential properties, including frame width, frame height, FourCC code (four-character code used to specify the video codec), and frames per second (FPS).

**Video Format Finder**

This script is designed to determine the format of the video frames in a given video file. 
It provides two solutions: one using OpenCV and the other using ffmpeg for Python.

---
## FrameEx.py

This script extracts specific video frames from a given video file based on the provided timestamps. 

---

## CalibrationCamera.py

This script is designed to calibrate a camera using a chessboard pattern. The camera's intrinsic parameters, such as the camera matrix and distortion coefficients, are calculated and then saved to a file.

---

## AudioDel.py
 The script removes the audio track from the input video with moviepy lib and saves a new video file without audio.

---

## Requirements

- Python 3.x
- OpenCV (cv2)
- numpy 
- MoviePy library