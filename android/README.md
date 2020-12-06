# 시각장애인을 위한 옷,색 탐지 및 추천 

### Overview

최근Deep Leaning 등 AI 기반 다양한 시스템과 어플리케이션이 많이 서비스 되고 있다. 

그 서비스들 중 많은 수가 Computer Vision 기술에서 문자 인식, 물체 감지 등을 이용한 다양한 어플리케이션을 개발 및 서비스 중이다. 

그러한 서비스들 중에는 사회적 약자계층을 위한 복지 서비스 또한 많으며, 정부 차원에서도 많은 지원을 통해 AI를 이용한 복지 서비스를 개발하고자 한다. 

본 논문에서는 시각장애인들의 의류 선택 도움을 위하여 물체 감지(Object detection) 기술을 이용한 의류 감지와 머신러닝 기법을 이용한 색감지를 기반으로한 의류 및 색상 추천 서비스를 개발하였다.

## Build the demo using Android Studio

### Prerequisites

*   If you don't have already, install
    **[Android Studio](https://developer.android.com/studio/index.html)**,
    following the instructions on the website.

*   You need an Android device and Android development environment with minimum
    API 21.

*   Android Studio 3.2 or later.

### Building

*   Open Android Studio, and from the Welcome screen, select Open an existing
    Android Studio project.

*   From the Open File or Project window that appears, navigate to and select
    the tensorflow-lite/examples/object_detection/android directory from
    wherever you cloned the TensorFlow Lite sample GitHub repo. Click OK.

*   If it asks you to do a Gradle Sync, click OK.

*   You may also need to install various platforms and tools, if you get errors
    like "Failed to find target with hash string 'android-21'" and similar.
    Click the `Run` button (the green arrow) or select `Run > Run 'android'`
    from the top menu. You may need to rebuild the project using `Build >
    Rebuild` Project.

*   This object detection Android reference app demonstrates two implementation
    solutions,
    [lib_task_api](https://github.com/tensorflow/examples/tree/master/lite/examples/nl_classification/android/lib_task_api)
    that leverages the out-of-box API from the
    [TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/object_detector),
    and
    [lib_interpreter](https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android/lib_interpreter)
    that creates the custom inference pipleline using the
    [TensorFlow Lite Interpreter Java API](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_java).
    You can change the build variant to whichever one you want to build and
    run—just go to `Build > Select Build Variant` and select one from the
    drop-down menu. See
    [configure product flavors in Android Studio](https://developer.android.com/studio/build/build-variants#product-flavors)
    for more details.

*   If it asks you to use Instant Run, click Proceed Without Instant Run.

*   Also, you need to have an Android device plugged in with developer options
    enabled at this point. See
    **[here](https://developer.android.com/studio/run/device)** for more details
    on setting up developer devices.

### Model used
Downloading, extraction and placing it in assets folder has been managed automatically by download.gradle.

If you explicitly want to download the model, you can download from **[here](http://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip)**. Extract the zip to get the .tflite and label file.


### Custom model used
This example shows you how to perform TensorFlow Lite object detection using a custom model.
* Clone the TensorFlow models GitHub repository to your computer.
```
git clone https://github.com/tensorflow/models/
```
* Build and install this repository.
```
cd models/research
python3 setup.py build && python3 setup.py install
```
* Download the MobileNet SSD trained on **[Open Images v4](https://storage.googleapis.com/openimages/web/factsfigures_v4.html)** **[here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)**. Extract the pretrained TensorFlow model files.
* Go to `models/research` directory and execute this code to get the frozen TensorFlow Lite graph.
```
python3 object_detection/export_tflite_ssd_graph.py \
  --pipeline_config_path object_detection/samples/configs/ssd_mobilenet_v2_oid_v4.config \
  --trained_checkpoint_prefix <directory with ssd_mobilenet_v2_oid_v4_2018_12_12>/model.ckpt \
  --output_directory exported_model
```
* Convert the frozen graph to the TFLite model.
```
tflite_convert \
  --input_shape=1,300,300,3 \
  --input_arrays=normalized_input_image_tensor \
  --output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 \
  --allow_custom_ops \
  --graph_def_file=exported_model/tflite_graph.pb \
  --output_file=<directory with the TensorFlow examples repository>/lite/examples/object_detection/android/app/src/main/assets/detect.tflite
```
`input_shape=1,300,300,3` because the pretrained model works only with that input shape.

`allow_custom_ops` is necessary to allow TFLite_Detection_PostProcess operation.

`input_arrays` and `output_arrays` can be drawn from the visualized graph of the example detection model.
```
bazel run //tensorflow/lite/tools:visualize \
  "<directory with the TensorFlow examples repository>/lite/examples/object_detection/android/app/src/main/assets/detect.tflite" \
  detect.html
```

* Get `labelmap.txt` from the second column of **[class-descriptions-boxable](https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv)**.
* In `DetectorActivity.java` set `TF_OD_API_IS_QUANTIZED` to `false`.


### Additional Note
_Please do not delete the assets folder content_. If you explicitly deleted the files, then please choose *Build*->*Rebuild* from menu to re-download the deleted model files into assets folder.
