# ML Kit Face Detection Sample App

The whole original work is in https://github.com/googlesamples/mlkit/tree/master/android/vision-quickstart

## Introduction

I only extracted one feature from the Vision Quickstart Sample App.
I deleted codes that are not used in Face Detection, and set the main activity to LivePreviewActivity.java.

* [Face Detection](https://developers.google.com/ml-kit/vision/face-detection/android) - Detect faces in real time and static images

## Getting Started

* Run the sample code on your Android device or emulator
* Try extending the code to add new features and functionality

## How to use the app

This app supports three usage scenarios: Live Camera

### Live Camera scenario
It uses the camera preview as input and contains the API workflow: Face Detection. 

There's also a settings page that allows you to configure several options:

* Camera
    * Preview size - Specify the preview size of rear/front camera manually (Default size is chosen appropriately based on screen size)
    * Enable live viewport - Toggle between blocking camera preview by API processing and result rendering or not
* Face Detection
    * Landmark mode -- Toggle between showing no or all facial landmarks
    * Contour mode -- Toggle between showing no or all contours
    * Classification mode -- Toggle between showing no or all classifications (smiling, eyes open/closed)
    * Performance mode -- Toggle between two operating modes (Fast or Accurate)
    * Face tracking -- Enable or disable face tracking
    * Minimum face size -- Choose the proportion of the head width to the image width

## Support

* [Documentation](https://developers.google.com/ml-kit/guides)
* [API Reference](https://developers.google.com/ml-kit/reference/android)
* [Stack Overflow](https://stackoverflow.com/questions/tagged/google-mlkit)


# Android-FaceRecognition
