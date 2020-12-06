
package org.tensorflow.lite.examples.detection.tflite;

import static java.lang.Math.min;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;
import android.util.Log;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.metadata.MetadataExtractor;


public class TFLiteObjectDetectionAPIModel implements Detector {
  private static final String TAG = "API_INTER";

  // Only return this many results.
  private static final int NUM_DETECTIONS = 10;
  // Float model
  private static final float IMAGE_MEAN = 127.5f;
  private static final float IMAGE_STD = 127.5f;
  // Number of threads in the java app
  private static final int NUM_THREADS = 4;
  private boolean isModelQuantized;
  // Config values.
  private int inputSize;
  // Pre-allocated buffers.
  private final List<String> labels = new ArrayList<>();
  private int[] intValues;
  // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
  // contains the location of detected boxes
  private float[][][] outputLocations;
  // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the classes of detected boxes
  private float[][] outputClasses;
  // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the scores of detected boxes
  private float[][] outputScores;
  // numDetections: array of shape [Batchsize]
  // contains the number of detected boxes
  private float[] numDetections;

  private float[][][] temp_buffer; //임시 저장

  private ByteBuffer imgData;

  private MappedByteBuffer tfLiteModel;
  private Interpreter.Options tfLiteOptions;
  private Interpreter tfLite;

  private TFLiteObjectDetectionAPIModel() {}

  /** Memory-map the model file in Assets. */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
      throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param modelFilename The model file path relative to the assets folder
   * @param labelFilename The label file path relative to the assets folder
   * @param inputSize The size of image input
   * @param isQuantized Boolean representing model is quantized or not
   */
  public static Detector create(
      final Context context,
      final String modelFilename,
      final String labelFilename,
      final int inputSize,
      final boolean isQuantized)
      throws IOException {
    final TFLiteObjectDetectionAPIModel d = new TFLiteObjectDetectionAPIModel();

    MappedByteBuffer modelFile = loadModelFile(context.getAssets(), modelFilename);
    MetadataExtractor metadata = new MetadataExtractor(modelFile);
    try (BufferedReader br =
        new BufferedReader(
            new InputStreamReader(
                metadata.getAssociatedFile(labelFilename), Charset.defaultCharset()))) {
      String line;
      while ((line = br.readLine()) != null) {
        Log.w(TAG, line);
        d.labels.add(line);
      }
    }

    d.inputSize = inputSize;

    try {
      Interpreter.Options options = new Interpreter.Options();
      options.setNumThreads(NUM_THREADS);
      d.tfLite = new Interpreter(modelFile, options);
      d.tfLiteModel = modelFile;
      d.tfLiteOptions = options;
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    d.isModelQuantized = isQuantized;
    // Pre-allocate buffers.
    int numBytesPerChannel;
    if (isQuantized) {
      numBytesPerChannel = 1; // Quantized
    } else {
      numBytesPerChannel = 4; // Floating point
    }
    d.imgData = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
    d.imgData.order(ByteOrder.nativeOrder());
    d.intValues = new int[d.inputSize * d.inputSize];

    d.outputLocations = new float[1][NUM_DETECTIONS][4];
    d.outputClasses = new float[1][NUM_DETECTIONS];
    d.outputScores = new float[1][NUM_DETECTIONS];
    d.numDetections = new float[1];
    return d;
  }

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    imgData.rewind();
    for (int i = 0; i < inputSize; ++i) {
      for (int j = 0; j < inputSize; ++j) {
        int pixelValue = intValues[i * inputSize + j];
        if (isModelQuantized) {
          // Quantized model
          imgData.put((byte) ((pixelValue >> 16) & 0xFF));
          imgData.put((byte) ((pixelValue >> 8) & 0xFF));
          imgData.put((byte) (pixelValue & 0xFF));
        } else { // Float model
          imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        }
      }
    }
    Trace.endSection(); // preprocessBitmap

    // Copy the input data into TensorFlow.
    Trace.beginSection("feed");
    outputLocations = new float[1][NUM_DETECTIONS][4];
    outputClasses = new float[1][NUM_DETECTIONS];
    outputScores = new float[1][NUM_DETECTIONS];
    numDetections = new float[1];

    temp_buffer = new float[1][100][7];

    Object[] inputArray = {imgData};
    Map<Integer, Object> outputMap = new HashMap<>();
    // for efficientNet
        outputMap.put(0, temp_buffer);

    //for mobileNet
//    outputMap.put(0,outputLocations);
//    outputMap.put(1,outputClasses);
//    outputMap.put(2,outputScores);
//    outputMap.put(3,numDetections);

    Trace.endSection();

    // Run the inference call.
    Trace.beginSection("run");
    tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
    Trace.endSection();

//for efficient
    for(int i=0; i<NUM_DETECTIONS; ++i){
      outputLocations[0][i][0]=temp_buffer[0][i][1];
      outputLocations[0][i][1]=temp_buffer[0][i][2];
      outputLocations[0][i][2]=temp_buffer[0][i][4];
      outputLocations[0][i][3]=temp_buffer[0][i][3];

      outputClasses[0][i] = temp_buffer[0][i][6];
      outputScores[0][i] = temp_buffer[0][i][5];
    }

    Log.i("Detect", Float.toString(outputClasses[0][0])+" "+Float.toString(outputScores[0][0]));

    numDetections[0]=NUM_DETECTIONS;

    // Show the best detections.
    // after scaling them back to the input size.
    // You need to use the number of detections from the output and not the NUM_DETECTONS variable
    // declared on top
    // because on some models, they don't always output the same total number of detections
    // For example, your model's NUM_DETECTIONS = 20, but sometimes it only outputs 16 predictions
    // If you don't use the output's numDetections, you'll get nonsensical data
    int numDetectionsOutput =
        min(
            NUM_DETECTIONS,
            (int) numDetections[0]); // cast from float to integer, use min for safety
    Log.d("Num_det", Float.toString(numDetections[0]));
    Log.d("Num_det_out", Integer.toString(NUM_DETECTIONS));
    Log.d("Num_det_out", Integer.toString(temp_buffer[0].length));

    final ArrayList<Recognition> recognitions = new ArrayList<>(numDetectionsOutput);
    //for efficientNet
        for (int i = 0; i < NUM_DETECTIONS; ++i) {
//          Log.d("inside for", Integer.toString(i));

          final RectF detection =
          new RectF(
              outputLocations[0][i][1],
              outputLocations[0][i][0],
              outputLocations[0][i][3],
              outputLocations[0][i][2]);

          if (outputClasses[0][i] > 39)
            continue;
      recognitions.add(
          new Recognition("" + i, labels.get((int) outputClasses[0][i]), outputScores[0][i], detection));
    }
//    for (int i = 0; i < NUM_DETECTIONS; ++i) {
//      final RectF detection =
//              new RectF(
//                      outputLocations[0][i][1] ,
//                      outputLocations[0][i][0] ,
//                      outputLocations[0][i][3] ,
//                      outputLocations[0][i][2] );
//
//      recognitions.add(
//              new Recognition(
//                      "" + i, labels.get((int) outputClasses[0][i]), outputScores[0][i], detection));
//    }
    Trace.endSection(); // "recognizeImage"
    return recognitions;
  }

  @Override
  public void enableStatLogging(final boolean logStats) {}

  @Override
  public String getStatString() {
    return "";
  }

  @Override
  public void close() {
    if (tfLite != null) {
      tfLite.close();
      tfLite = null;
    }
  }

  @Override
  public void setNumThreads(int numThreads) {
    if (tfLite != null) {
      tfLiteOptions.setNumThreads(numThreads);
      recreateInterpreter();
    }
  }

  @Override
  public void setUseNNAPI(boolean isChecked) {
    if (tfLite != null) {
      tfLiteOptions.setUseNNAPI(isChecked);
      recreateInterpreter();
    }
  }

  private void recreateInterpreter() {
    tfLite.close();
    tfLite = new Interpreter(tfLiteModel, tfLiteOptions);
  }


  //load library for openCV
  static {
    System.loadLibrary("opencv_java4");
  }


}
