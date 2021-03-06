# Android-FaceRecognition

This project is based on Firebase ML Kit [Face Detection](https://developers.google.com/ml-kit/vision/face-detection/android).

The face recognition model used in this example is [JHFace](https://github.com/joonb14/JHFace). Which is heavily based on [arcface-tf2](https://github.com/peteryuX/arcface-tf2) implementation.

 ![demo](./demo/demo.gif)

## Details

This project is trained on MS-Celeb-1M dataset, using MobileNetV2 as backbone, ArcFace for loss function. Then used IJB-C dataset for extracting centroid for every subjects, applied cosine similarity for face matching.

## Tutorials

I'm now working on uploading tutorials on making this project. Already wrote it down in markdown in Korean on my [github page](https://joonb14.github.io/tag/face-recognition/) I'll provide English version soon.


