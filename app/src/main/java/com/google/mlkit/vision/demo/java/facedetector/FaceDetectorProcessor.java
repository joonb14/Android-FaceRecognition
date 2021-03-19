/*
 * Copyright 2020 Google LLC. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.mlkit.vision.demo.java.facedetector;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.PointF;
import androidx.annotation.NonNull;

import android.graphics.Rect;
import android.util.Log;
import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.demo.GraphicOverlay;
import com.google.mlkit.vision.demo.java.CosineSim;
import com.google.mlkit.vision.demo.java.VisionProcessorBase;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import com.google.mlkit.vision.face.FaceLandmark;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.PriorityQueue;

import androidx.renderscript.Allocation;
import androidx.renderscript.RenderScript;

//TF Lite
import org.tensorflow.lite.Interpreter;

/** Face Detector Demo. */
public class FaceDetectorProcessor extends VisionProcessorBase<List<Face>> {

  private static final String TAG = "FaceDetectorProcessor";

  /** member variables **/
  public static Bitmap image;
  private float[][][][] face_input;
  private RenderScript RS;
  private ScriptC_singlesource script;
  public static Interpreter tflite;
  private Float[][] centroid;
  private String[] labels;

  /** config values **/
  private final boolean THREE_CHANNEL = true; // false for Black and White image, true for RGB image

  private final FaceDetector detector;

  public FaceDetectorProcessor(Context context) {
    this(
        context,
        new FaceDetectorOptions.Builder()
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .enableTracking()
            .build());
  }

  public FaceDetectorProcessor(Context context, FaceDetectorOptions options) {
    super(context);
    Log.v(MANUAL_TESTING_LOG, "Face detector options: " + options);
    detector = FaceDetection.getClient(options);
    RS = RenderScript.create(context);
    script = new ScriptC_singlesource(RS);

    //Load Centroid Vectors and labels
    try {
      //Centroid
      InputStream inputStream = context.getAssets().open("centroid_list.csv");
      BufferedReader br = new BufferedReader(new InputStreamReader(inputStream));
      List<List<Float>> centroid_list = new ArrayList<List<Float>>();
      String line = "";

      while((line = br.readLine()) != null){
        List<Float> tmpList = new ArrayList<Float>();
        String[] array = line.split(",");
        Float[] floats = Arrays.stream(array).map(Float::valueOf).toArray(Float[]::new);
        tmpList = Arrays.asList(floats);
        centroid_list.add(tmpList);
      }
      centroid = centroid_list.stream()
              .map(l -> l.toArray(new Float[0]))
              .toArray(Float[][]::new);
      br.close();

      //Labels
      inputStream = context.getAssets().open("id_list.txt");
      br = new BufferedReader(new InputStreamReader(inputStream));
      List<String> label_list = new ArrayList<String>();

      while((line = br.readLine()) != null){
        label_list.add(line);
      }
      labels = new String[label_list.size()];
      label_list.toArray(labels);
      br.close();
    }
    catch (IOException e) {
      e.printStackTrace();
    }
  }

  @Override
  public void stop() {
    super.stop();
    detector.close();
  }

  @Override
  protected Task<List<Face>> detectInImage(InputImage image) {
    return detector.process(image);
  }

  @Override
  protected void onSuccess(@NonNull List<Face> faces, @NonNull GraphicOverlay graphicOverlay) {
    for (Face face : faces) {

      /** Extract Face Bitmap **/
      int resolution = 112;
      Rect facePos = face.getBoundingBox();
      float faceboxWsize = facePos.right - facePos.left;
      float faceboxHsize = facePos.bottom - facePos.top;
      float[] mirrorY = {
              -1, 0, 0,
              0, 1, 0,
              0, 0, 1
      };
      try {
        Matrix matrix = new Matrix();
        matrix.setValues(mirrorY);
        Bitmap faceBitmap = Bitmap.createBitmap(
                image,
                (int) facePos.left, (int) facePos.top,
                (int) faceboxWsize, (int) faceboxHsize,
                matrix,
                false);
        faceBitmap = Bitmap.createScaledBitmap(faceBitmap, resolution, resolution, false);

        //Renderscript converts RGBA value to YUV's Y value.
        //After RenderScript, Y value will be stored in Red pixel value
        if (!THREE_CHANNEL) {
          Allocation inputAllocation = Allocation.createFromBitmap(RS, faceBitmap);
          Allocation outputAllocation = Allocation.createTyped(RS, inputAllocation.getType());
          script.invoke_process(inputAllocation, outputAllocation);
          outputAllocation.copyTo(faceBitmap);
        }

        int[] face_pix = new int[resolution * resolution];
        faceBitmap.getPixels(face_pix, 0, resolution, 0, 0, resolution, resolution);
        face_input = new float[1][resolution][resolution][3];
        for (int y = 0; y < resolution; y++) {
          for (int x = 0; x < resolution; x++) {
            int index = y * resolution + x;
            face_input[0][y][x][0] = ((face_pix[index] & 0x00FF0000) >> 16) / (float) 255.0f;
            face_input[0][y][x][1] = ((face_pix[index] & 0x0000FF00) >> 8) / (float) 255.0f;
            face_input[0][y][x][2] = (face_pix[index] & 0x000000FF) / (float) 255.0f;
          }
        }

        // For Single input
        float[][] output = new float[1][512];
        tflite.run(face_input, output);

        // To use multiple input and multiple output you must use the Interpreter.runForMultipleInputsOutputs()
//      float[][][][][] inputs = new float[][][][][]{face_input};
//      float[][] output = new float[1][512];
//      Map<Integer, Object> outputs = new HashMap<>();
//      outputs.put(0, output);
//      //Run TFLite
//      tflite.runForMultipleInputsOutputs(inputs, outputs);
//      Log.d(TAG,"Output[0][0]:"+output[0][0]);

        // Compute Cosine Similarity
        PriorityQueue<CosineSim> cosine_sim = new PriorityQueue<>();
        for (int i=0;i< centroid.length; i++) {
          float cosine = cosineSimilarity(centroid[i], output[0]);
          cosine_sim.offer(new CosineSim(labels[i],cosine));
        }

        CosineSim [] top5_list = new CosineSim[5];
        Log.d(TAG, "--------------------------------split--------------------------------");
        for (int i=0;i<5;i++) {
          top5_list[i] = cosine_sim.poll();
          Log.d(TAG, "Top"+i+" Cosine similarity: " + top5_list[i].toString());
        }

        graphicOverlay.add(new FaceGraphic(graphicOverlay, face, top5_list));
        logExtrasForTesting(face);
      }
      catch (java.lang.IllegalArgumentException e) {
        Log.e(TAG, "java.lang.IllegalArgumentException");
        e.printStackTrace();
      }
    }
  }

  private static float cosineSimilarity(Float[] vectorA, float[] vectorB) {
    double dotProduct = 0.0;
    double normA = 0.0;
    double normB = 0.0;
    for (int i = 0; i < vectorA.length; i++) {
      dotProduct += vectorA[i] * vectorB[i];
      normA += Math.pow(vectorA[i], 2);
      normB += Math.pow(vectorB[i], 2);
    }
    return (float) (dotProduct / (Math.sqrt(normA) * Math.sqrt(normB)));
  }

  private static void logExtrasForTesting(Face face) {
    if (face != null) {
      Log.v(MANUAL_TESTING_LOG, "face bounding box: " + face.getBoundingBox().flattenToString());
      Log.v(MANUAL_TESTING_LOG, "face Euler Angle X: " + face.getHeadEulerAngleX());
      Log.v(MANUAL_TESTING_LOG, "face Euler Angle Y: " + face.getHeadEulerAngleY());
      Log.v(MANUAL_TESTING_LOG, "face Euler Angle Z: " + face.getHeadEulerAngleZ());

      // All landmarks
      int[] landMarkTypes =
          new int[] {
            FaceLandmark.MOUTH_BOTTOM,
            FaceLandmark.MOUTH_RIGHT,
            FaceLandmark.MOUTH_LEFT,
            FaceLandmark.RIGHT_EYE,
            FaceLandmark.LEFT_EYE,
            FaceLandmark.RIGHT_EAR,
            FaceLandmark.LEFT_EAR,
            FaceLandmark.RIGHT_CHEEK,
            FaceLandmark.LEFT_CHEEK,
            FaceLandmark.NOSE_BASE
          };
      String[] landMarkTypesStrings =
          new String[] {
            "MOUTH_BOTTOM",
            "MOUTH_RIGHT",
            "MOUTH_LEFT",
            "RIGHT_EYE",
            "LEFT_EYE",
            "RIGHT_EAR",
            "LEFT_EAR",
            "RIGHT_CHEEK",
            "LEFT_CHEEK",
            "NOSE_BASE"
          };
      for (int i = 0; i < landMarkTypes.length; i++) {
        FaceLandmark landmark = face.getLandmark(landMarkTypes[i]);
        if (landmark == null) {
          Log.v(
              MANUAL_TESTING_LOG,
              "No landmark of type: " + landMarkTypesStrings[i] + " has been detected");
        } else {
          PointF landmarkPosition = landmark.getPosition();
          String landmarkPositionStr =
              String.format(Locale.US, "x: %f , y: %f", landmarkPosition.x, landmarkPosition.y);
          Log.v(
              MANUAL_TESTING_LOG,
              "Position for face landmark: "
                  + landMarkTypesStrings[i]
                  + " is :"
                  + landmarkPositionStr);
        }
      }
      Log.v(
          MANUAL_TESTING_LOG,
          "face left eye open probability: " + face.getLeftEyeOpenProbability());
      Log.v(
          MANUAL_TESTING_LOG,
          "face right eye open probability: " + face.getRightEyeOpenProbability());
      Log.v(MANUAL_TESTING_LOG, "face smiling probability: " + face.getSmilingProbability());
      Log.v(MANUAL_TESTING_LOG, "face tracking id: " + face.getTrackingId());
    }
  }

  @Override
  protected void onFailure(@NonNull Exception e) {
    Log.e(TAG, "Face detection failed " + e);
  }
}
