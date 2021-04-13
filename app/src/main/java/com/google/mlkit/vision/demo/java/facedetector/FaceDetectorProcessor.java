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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
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
  private RenderScript RS;
  private ScriptC_singlesource script;
  public static Interpreter tflite;
  private Float[][] centroid;
  private String[] labels;
  private ByteBuffer imgData; // for Qunatized model

  /** config values **/
  private final boolean THREE_CHANNEL = true; // false for Black and White image, true for RGB image
  private final boolean QUANTIZATION = true;
  // Float model
  private static final float IMAGE_MEAN = 127.5f;
  private static final float IMAGE_STD = 127.5f;
  private static final int inputSize = 112;
  private static final boolean isDebug = true;
  private static final boolean isRankN = true;

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
    int numBytesPerChannel;
    if (QUANTIZATION) {
      numBytesPerChannel = 1; // Quantized
    } else {
      numBytesPerChannel = 4; // Floating point
    }
    imgData = ByteBuffer.allocateDirect(inputSize * inputSize * 3 * numBytesPerChannel);
    imgData.order(ByteOrder.nativeOrder());

    //Load Centroid Vectors and labels
    try {
      //Centroid
      InputStream inputStream;
      if (QUANTIZATION) {
        inputStream = context.getAssets().open("quant_centroid_list.csv");
      }
      else {
        inputStream = context.getAssets().open("centroid_list.csv");
      }
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
      if (QUANTIZATION) {
        inputStream = context.getAssets().open("quant_id_list.txt");
      }
      else {
        inputStream = context.getAssets().open("id_list.txt");
      }
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
      if(isDebug) Log.d("LATENCY CHECK","Extract Face Bitmap");
      int resolution = inputSize;
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

        if(isDebug) Log.d("LATENCY CHECK","Input preprocessing");
        //Renderscript converts RGBA value to YUV's Y value.
        //After RenderScript, Y value will be stored in Red pixel value
        if (!THREE_CHANNEL) {
          Allocation inputAllocation = Allocation.createFromBitmap(RS, faceBitmap);
          Allocation outputAllocation = Allocation.createTyped(RS, inputAllocation.getType());
          script.invoke_process(inputAllocation, outputAllocation);
          outputAllocation.copyTo(faceBitmap);
        }

        imgData.rewind();
        int[] face_pix = new int[resolution * resolution];
        faceBitmap.getPixels(face_pix, 0, resolution, 0, 0, resolution, resolution);
        for (int y = 0; y < resolution; y++) {
          for (int x = 0; x < resolution; x++) {
            int index = y * resolution + x;
            if (QUANTIZATION) {
              imgData.put((byte) ((face_pix[index] >> 16) & 0xFF));
              imgData.put((byte) ((face_pix[index] >> 8) & 0xFF));
              imgData.put((byte) (face_pix[index] & 0xFF));
            }
            else {
              // float model
              imgData.putFloat((((face_pix[index] >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
              imgData.putFloat((((face_pix[index] >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
              imgData.putFloat(((face_pix[index] & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
            }
          }
        }

        if(isDebug) Log.d("LATENCY CHECK","Model Inference");
        byte[][] output_int = new byte[1][512];
        float[][] output = new float[1][512];
        // For Single input
        if (QUANTIZATION) {
          tflite.run(imgData, output_int);
        }
        else {
          tflite.run(imgData, output);
        }

        if(isDebug) Log.d("LATENCY CHECK","Output Normalization");
        float [] norm_out = new float[512];
        if (QUANTIZATION){
          double norm  = norm(output_int[0]);
          for (int i=0;i< output_int[0].length;i++){
            int uint8 = output_int[0][i] & 0xFF;
            norm_out[i] = (float)(uint8/norm);
          }
        }
        else {
          double norm  = norm(output[0]);
          for (int i=0;i< output[0].length;i++){
            norm_out[i] = (float)(output[0][i]/norm);
          }
        }
        CosineSim[] top5_list;
        if(isRankN) {
          // Compute Cosine Similarity
          if(isDebug) Log.d("LATENCY CHECK","Compute Cosine Similarity");
          PriorityQueue<CosineSim> cosine_sim = new PriorityQueue<>();
          for (int i=0;i< centroid.length; i++) {
            float cosine = cosineSimilarity(centroid[i], norm_out);
            cosine_sim.offer(new CosineSim(labels[i],cosine));
          }
          top5_list = new CosineSim[5];
          if (isDebug) Log.d("LATENCY CHECK", "Result");
          Log.d(TAG, "--------------------------------Result--------------------------------");
          for (int i = 0; i < 5; i++) {
            top5_list[i] = cosine_sim.poll();
            Log.d(TAG, "Top" + i + " Cosine similarity: " + top5_list[i].toString());
          }
        }
        else{
          // Compute Cosine Similarity
          if(isDebug) Log.d("LATENCY CHECK","Compute Cosine Similarity");
          float max = 0;
          String label = "";
          for (int i=0;i< centroid.length; i++) {
            float cosine = cosineSimilarity(centroid[i], norm_out);
            if (cosine > max) {
              label = labels[i];
              max = cosine;
            }
          }
          top5_list = new CosineSim[1];
          top5_list[0] = new CosineSim(label,max);
        }
        if(isDebug) Log.d("LATENCY CHECK","graphicOverlay");
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

  public static double norm(byte[] data) {
    return (Math.sqrt(sumSquares(data)));
  }
  public static double norm(float[] data) {
    return (Math.sqrt(sumSquares(data)));
  }
  public static int sumSquares(byte[] data) {
    int ans = 0;
    for (int k = 0; k < data.length; k++) {
      int uint8 = data[k] & 0xFF;
      ans += uint8 * uint8;
    }
    return (ans);
  }
  public static float sumSquares(float[] data) {
    float ans = 0.0f;
    for (int k = 0; k < data.length; k++) {
      ans += data[k] * data[k];
    }
    return (ans);
  }
}
