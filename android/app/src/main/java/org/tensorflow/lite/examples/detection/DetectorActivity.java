package org.tensorflow.lite.examples.detection;

import android.app.AlertDialog;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.graphics.drawable.ColorDrawable;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.os.SystemClock;
import android.os.Vibrator;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.View;
import android.widget.ProgressBar;
import android.widget.Toast;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Detector;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

import static org.opencv.core.CvType.CV_8UC3;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener, View.OnClickListener {
    private static final Logger LOGGER = new Logger();


    public static final int sub = 1001; // ColorActivity에 쓰이는 요청코드(상수)

    // Configuration values for the prepackaged SSD model.
    //for efficientDet
    private static final int TF_OD_API_INPUT_SIZE = 512;
    //for mobileNet 640*640
//  private static final int TF_OD_API_INPUT_SIZE = 1280;
    private static final boolean TF_OD_API_IS_QUANTIZED = true;
    private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
    private static final String TF_OD_API_LABELS_FILE = "labelmap.txt";
    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    // Minimum detection confidence to track a detection.
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
    private static final boolean MAINTAIN_ASPECT = false;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;
    OverlayView trackingOverlay;
    private Integer sensorOrientation;

    private Detector detector;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;
    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private MultiBoxTracker tracker;

    private BorderedText borderedText;

    public String titles_to_pop;
    public float[] result_buffer;
    public Bitmap bitmap_buffer = null;
    public String colors_to_pop;
    public String[] recomd_string;
    public String tone;
    public String title_buffer;
//  public static Context context_main
    public Vibrator vib;

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);

        int cropSize = TF_OD_API_INPUT_SIZE;

        try {
            detector =
                    TFLiteObjectDetectionAPIModel.create(
                            this,
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_INPUT_SIZE,
                            TF_OD_API_IS_QUANTIZED);
            cropSize = TF_OD_API_INPUT_SIZE;
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing Detector!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Detector could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        tracker.draw(canvas);
                        if (isDebug()) {
                            tracker.drawDebug(canvas);
                        }
                    }
                });

        tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
    }

    @Override
    protected void processImage() {

        vib = (Vibrator)getSystemService(Context.VIBRATOR_SERVICE);

        ++timestamp;
        final long currTimestamp = timestamp;
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        LOGGER.i("Running detection on image " + currTimestamp);
                        final long startTime = SystemClock.uptimeMillis();
                        final List<Detector.Recognition> results = detector.recognizeImage(croppedBitmap);
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                        final Canvas canvas = new Canvas(cropCopyBitmap);
                        final Paint paint = new Paint();
                        paint.setColor(Color.RED);
                        paint.setStyle(Style.STROKE);
                        paint.setStrokeWidth(2.0f);

                        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                        switch (MODE) {
                            case TF_OD_API:
                                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                                break;
                        }

                        final List<Detector.Recognition> mappedRecognitions =
                                new ArrayList<Detector.Recognition>();

                        float thres = -1.0f;
                        for (final Detector.Recognition result : results) {
                            final RectF location = result.getLocation();
                            if (location != null && result.getConfidence() >= minimumConfidence) {
                                canvas.drawRect(location, paint);
                                if(title_buffer == null)
                                    vib.vibrate(250);
                                else if(!title_buffer.equals(result.getTitle()))
                                    vib.vibrate(250);
                                cropToFrameTransform.mapRect(location);

                                result.setLocation(location);
                                mappedRecognitions.add(result);
                                Log.d("title", result.getTitle());
                                if (thres < result.getConfidence()) {
                                    thres = result.getConfidence();
                                    title_buffer = result.getTitle();
                                    result_buffer = new float[]{result.getLocation().top, result.getLocation().left, result.getLocation().bottom, result.getLocation().right};
                                    bitmap_buffer = Bitmap.createBitmap(croppedBitmap);
                                }
                            }
                        }

                        tracker.trackResults(mappedRecognitions, currTimestamp);
                        trackingOverlay.postInvalidate();

                        computingDetection = false;

                        runOnUiThread(
                                new Runnable() {
                                    @Override
                                    public void run() {
                                        showFrameInfo(previewWidth + "x" + previewHeight);
                                        showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                                        showInference(lastProcessingTimeMs + "ms");
                                    }
                                });
                    }
                });
    }

    @Override
    protected int getLayoutId() {
        return R.layout.tfe_od_camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum DetectorMode {
        TF_OD_API;
    }

    @Override
    protected void setUseNNAPI(final boolean isChecked) {
        runInBackground(
                () -> {
                    try {
                        detector.setUseNNAPI(isChecked);
                    } catch (UnsupportedOperationException e) {
                        LOGGER.e(e, "Failed to set \"Use NNAPI\".");
                        runOnUiThread(
                                () -> {
                                    Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
                                });
                    }
                });
    }

    @Override
    protected void setNumThreads(final int numThreads) {
        runInBackground(
                () -> {
                    try {
                        detector.setNumThreads(numThreads);
                    } catch (IllegalArgumentException e) {
                        LOGGER.e(e, "Failed to set multithreads.");
                        runOnUiThread(
                                () -> {
                                    Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
                                });
                    }
                });
    }

    private Bitmap col_bit = null;
    private float[] detected_loc = null;
    Scalar extracted_col;

    String recommend1 = "";
    String recommend2 = "";
    String recommend3 = "";

    @Override
    public void onClick(View v) {
        if (v.getId() == R.id.cloth_recommend) {
//      String arr = ((DetectorActivity)DetectorActivity.context_main).titles_to_pop;
            if(title_buffer != null){
                titles_to_pop = title_buffer;
                col_bit = bitmap_buffer;
                detected_loc = result_buffer;
                new AlertDialog.Builder(DetectorActivity.this).setTitle("옷 감지").setMessage("감지된 옷은 " + titles_to_pop + " 입니다.")
                        .setNeutralButton("닫기", new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialog, int which) {

                            }
                        }).show();
                title_buffer = null;
            } else {
                new AlertDialog.Builder(DetectorActivity.this).setTitle("옷 감지 실패").setMessage("감지된 옷이 없습니다.\n\n")
                        .setNeutralButton("닫기", new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialog, int which) {

                            }
                        })
                        .show();
            }
        } else if (v.getId() == R.id.color_recommend) {
            Log.d("recom", Boolean.toString(col_bit != null) +  Boolean.toString(detected_loc != null) + Boolean.toString(titles_to_pop != null));
            if (col_bit != null && detected_loc != null && titles_to_pop != null) {


                ProgressDialog progressDialog = new ProgressDialog(DetectorActivity.this);
                progressDialog.setMessage("추천중");
                progressDialog.setCancelable(true);
                progressDialog.show();

                Handler handler = new Handler(){
                    public void handleMessage(Message msg){
                        super.handleMessage(msg);
                        switch(msg.what) {
                            default:
                                progressDialog.dismiss();
                        }
                    }
                };

                Thread getcol = new Thread(new Runnable() {
                    public void run() {

                        Log.d("col recom", "ext col");
                        extracted_col = extract_col_from_detection(col_bit, detected_loc);

                        Log.d("col recom", "recomd col : " + extracted_col.toString());
                        //color recommandation

                        double saturation = extracted_col.val[1];//채도
                        double value = extracted_col.val[2];//명도
                        double hue = extracted_col.val[0]; //색

                        double ran = (100 - 25) / 2;

                        double[] s_sat = new double[5];
                        double[] s_val = new double[5];
                        double[] s_hue = new double[5];

                        if (saturation <= 25) {
                            if (value > 50)
                                colors_to_pop = "흰색";
                            else if (value <= 50 && value > 25) {
                                if (value > 40)
                                    colors_to_pop = "연한 회색";
                                else
                                    colors_to_pop = "회색";
                            }
                            tone = "";
                        }

                        if (value <= 25) {
                            colors_to_pop = "검은색";
                            tone = "";
                        }
                        if (saturation > 25 && value > 25) {
                            if (hue > 10 && hue <= 47) {
                                colors_to_pop = "주황색";
                                if (saturation > ran) {
                                    if (value > ran)
                                        tone = "비비드톤";
                                    else {
                                        tone = "딥톤";
                                        colors_to_pop = "갈색";//갈색느낌
                                    }
                                } else {
                                    if (value > ran)
                                        tone = "파스텔톤";
                                    else {
                                        tone = "모노톤";
                                        colors_to_pop = "갈색";
                                    }
                                }
                            } else if (hue > 47 && hue <= 70) {
                                colors_to_pop = "노란색";
                                if (saturation > ran) {
                                    if (value > ran)
                                        tone = "비비드톤";
                                    else
                                        tone = "딥톤";
                                } else {
                                    if (value > ran)
                                        tone = "파스텔톤";
                                    else
                                        tone = "모노톤";
                                }
                            } else if (hue > 70 && hue <= 120) {
                                colors_to_pop = "연두색";
                                if (saturation > ran) {
                                    if (value > ran)
                                        tone = "비비드톤";
                                    else
                                        tone = "딥톤";
                                } else {
                                    if (value > ran)
                                        tone = "파스텔톤";
                                    else
                                        tone = "모노톤";
                                }
                            } else if (hue > 120 && hue <= 165) {
                                colors_to_pop = "초록색";
                                if (saturation > ran) {
                                    if (value > ran)
                                        tone = "비비드톤";
                                    else {
                                        tone = "딥톤";
                                        colors_to_pop = "카키색";//카키느낌
                                    }
                                } else {
                                    if (value > ran)
                                        tone = "파스텔톤";
                                    else {
                                        tone = "모노톤";
                                        colors_to_pop = "카키색";
                                    }
                                }
                            } else if (hue > 165 && hue <= 200) {
                                colors_to_pop = "하늘색";
                                if (saturation > ran) {
                                    if (value > ran)
                                        tone = "비비드톤";
                                    else
                                        tone = "딥톤";
                                } else {
                                    if (value > ran)
                                        tone = "파스텔톤";
                                    else
                                        tone = "모노톤";
                                }
                            } else if (hue > 200 && hue <= 255) {
                                colors_to_pop = "파란색";
                                if (saturation > ran) {
                                    if (value > ran)
                                        tone = "비비드톤";
                                    else {
                                        tone = "딥톤";
                                        colors_to_pop = "남색";//남색느낌
                                    }
                                } else {
                                    if (value > ran)
                                        tone = "파스텔톤";
                                    else
                                        tone = "모노톤";
                                }
                            } else if (hue > 255 && hue <= 300) {
                                colors_to_pop = "보라색";
                                if (saturation > ran) {
                                    if (value > ran)
                                        tone = "비비드톤";
                                    else
                                        tone = "딥톤";
                                } else {
                                    if (value > ran)
                                        tone = "파스텔톤";
                                    else
                                        tone = "모노톤";
                                }
                            } else if (hue > 300 && hue <= 345) {
                                colors_to_pop = "분홍색";
                                if (saturation > ran) {
                                    if (value > ran)
                                        tone = "비비드톤";
                                    else
                                        tone = "딥톤";
                                } else {
                                    if (value > ran)
                                        tone = "파스텔톤";
                                    else
                                        tone = "모노톤";
                                }
                            } else if ((hue > 345 && hue <= 360) || (hue <= 10)) {
                                colors_to_pop = "빨간색";
                                if (saturation > ran) {
                                    if (value > ran)
                                        tone = "비비드톤";
                                    else {
                                        tone = "딥톤";
                                        colors_to_pop = "와인색";
                                    }
                                } else {
                                    if (value > ran)
                                        tone = "파스텔톤";
                                    else {
                                        tone = "모노톤";
                                        colors_to_pop = "와인색";
                                    }
                                }
                            }
                        }

                        if (tone == "파스텔톤") {
                            if (colors_to_pop == "주황색")
                                colors_to_pop = "베이지색";
                            else if (colors_to_pop == "초록색")
                                colors_to_pop = "연두색";
                            else if (colors_to_pop == "파란색")
                                colors_to_pop = "하늘색";
                            else {
                                String text1 = "연한";
                                colors_to_pop = text1 + " " + colors_to_pop;
                            }
                        }

                        for (int i = 0; i < 2; i++){
                            if(saturation > ran)
                                s_sat[i] = saturation - 62.5;
                            else
                                s_sat[i] = 62.5 - saturation;
                            s_hue[i] = hue;

                            if(value > ran)
                                s_val[1] = saturation - 62.5;
                            else
                                s_val[1] = 62.5 - saturation;
                        }
                        s_val[0] = value;

                        for(int j = 2; j < 5; j++) {
                            s_sat[j] = saturation;
                            s_val[j] = value;
                        }

                        if(colors_to_pop.contains("흰색")) {
                            recomd_string = new String[]{"회색", "검은색", "하늘색", "노란색", "베이지"};
                        }
                        else if(colors_to_pop.contains("회색")) {
                            recomd_string = new String[]{"검은색", "와인색", "베이지"};
                        }
                        else if(colors_to_pop.contains("검은색")){
                            recomd_string= new String[]{"검은색", "흰색", "노란색", "갈색"};
                        }
                        else if(colors_to_pop.contains("주황색")){
                            recomd_string= new String[]{"검은색", "흰색","하늘색"};
                        }
                        else if(colors_to_pop.contains("갈색")){
                            recomd_string= new String[]{"회색", "카키색", "베이지"};
                        }
                        else if(colors_to_pop.contains("노란색")){
                            recomd_string= new String[]{"검은색", "흰색", "파란색"};
                        }
                        else if(colors_to_pop.contains("연두색")){
                            recomd_string= new String[]{"베이지색", "흰색", "갈색"};
                        }
                        else if(colors_to_pop.contains("초록색")){
                            recomd_string= new String[]{"검은색", "흰색", "베이지"};
                        }
                        else if(colors_to_pop.contains("카키색")){
                            recomd_string= new String[]{"검은색", "흰색", "파란색"};
                        }
                        else if(colors_to_pop.contains("하늘색")){
                            recomd_string= new String[]{"흰색", "남색", "노란색"};
                        }
                        else if(colors_to_pop.contains("파란색")){
                            recomd_string= new String[]{"흰색", "갈색", "베이지"};
                        }
                        else if(colors_to_pop.contains("남색")){
                            recomd_string= new String[]{"회색", "베이지", "검은색"};
                        }
                        else if(colors_to_pop.contains("보라색")){
                            recomd_string= new String[]{"회색", "네이비", "카키"};
                        }
                        else if(colors_to_pop.contains("분홍색")){
                            recomd_string= new String[]{"남색", "베이지", "흰색"};
                        }
                        else if(colors_to_pop.contains("빨간색")){
                            recomd_string= new String[]{"검은색", "흰색", "남색", "베이지"};
                        }
                        else if(colors_to_pop.contains("와인색")){
                            recomd_string= new String[]{"베이지", "하늘색", "회색"};
                        }
                        else if(colors_to_pop.contains("베이지")){
                            recomd_string= new String[]{"하늘색", "회색", "갈색"};
                        }

                        DetectorActivity.this.runOnUiThread(new Runnable() {
                            public void run() {

                                new AlertDialog.Builder(DetectorActivity.this).setMessage("성별을 골라주세요").setPositiveButton("남자",
                                        new DialogInterface.OnClickListener() {
                                            @Override
                                            public void onClick(DialogInterface dialog, int which) {
                                                if(titles_to_pop.equals("블레이저")){
                                                    recommend1 = "면바지";
                                                    recommend2 = "청바지";
                                                }
                                                else if(titles_to_pop.equals("항공점퍼")){
                                                    recommend1 = "면바지";
                                                    recommend2 = "청바지";
                                                    recommend3 = "7부바지";
                                                }
                                                else if(titles_to_pop.equals("가디건")){
                                                    recommend1 = "면바지";
                                                    recommend2 = "청바지";
                                                }
                                                else if(titles_to_pop.equals("파카")){
                                                    recommend1 = "청바지";
                                                    recommend2 = "면바지";
                                                }
                                                else if(titles_to_pop.equals("피코트")){
                                                    recommend1 = "면바지";
                                                    recommend2 = "청바지";
                                                }
                                                else if(titles_to_pop.equals("코트")){
                                                    recommend1 = "면바지";
                                                    recommend2 = "청바지";
                                                }
                                                else if(titles_to_pop.equals("자켓")){
                                                    recommend1 = "면바지";
                                                    recommend2 = "반바지";
                                                    recommend3 = "청바지";
                                                }
                                                // 상의
                                                else if(titles_to_pop.equals("아노락")){
                                                    recommend1 = "면바지";
                                                    recommend2 = "청바지";
                                                    recommend3 = "트레이닝복 바지";
                                                }
                                                else if(titles_to_pop.equals("셔츠")){
                                                    recommend1 = "청바지";
                                                    recommend2 = "반바지";
                                                    recommend3 = "면바지";
                                                }
                                                else if(titles_to_pop.equals("후드")){
                                                    recommend1 = "청바지";
                                                    recommend2 = "반바지";
                                                    recommend3 = "면바지";
                                                }
                                                else if(titles_to_pop.equals("스웨터")){
                                                    recommend1 = "면바지";
                                                    recommend2 = "청바지";
                                                }
                                                else if(titles_to_pop.equals("탱크탑")){
                                                    recommend1 = "반바지";
                                                    recommend2 = "트레이닝복 바지";
                                                    recommend3 = "조거";
                                                }
                                                else if(titles_to_pop.equals("티셔츠")){
                                                    recommend1 = "청바지";
                                                    recommend2 = "트레이닝복 바지";
                                                    recommend3 = "조거";
                                                }
                                                else if(titles_to_pop.equals("헨리넥")){
                                                    recommend1 = "청바지";
                                                    recommend2 = "조거";
                                                    recommend3 = "반바지";
                                                }
                                                else if(titles_to_pop.equals("터틀넥")){
                                                    recommend1 = "면바지";
                                                    recommend2 = "청바지";
                                                }
                                                //하의
                                                else if(titles_to_pop.equals("7부바지")){
                                                    recommend1 = "셔츠";
                                                    recommend2 = "티셔츠";
                                                    recommend3 = "후드";
                                                }
                                                else if(titles_to_pop.equals("면바지")){
                                                    recommend1 = "셔츠";
                                                    recommend2 = "블레이저";
                                                    recommend3 = "티셔츠";
                                                }
                                                else if(titles_to_pop.equals("청바지")){
                                                    recommend1 = "티셔츠";
                                                    recommend2 = "셔츠";
                                                    recommend3 = "자켓";
                                                }
                                                else if(titles_to_pop.equals("조거")){
                                                    recommend1 = "티셔츠";
                                                    recommend2 = "자켓";
                                                    recommend3 = "후드";
                                                }
                                                else if(titles_to_pop.equals("레깅스")){
                                                    recommend1 = "티셔츠";
                                                    recommend2 = "항공점퍼";
                                                    recommend3 = "후드";
                                                }
                                                else if(titles_to_pop.equals("반바지")){
                                                    recommend1 = "셔츠";
                                                    recommend2 = "블레이저";
                                                    recommend3 = "자켓";
                                                }
                                                else if(titles_to_pop.equals("트레이닝복바지")){
                                                    recommend1 = "자켓";
                                                    recommend2 = "티셔츠";
                                                    recommend3 = "후드";
                                                }

                                                String str1 = "추천옷은 ";
                                                if(recommend1 != null)
                                                    str1 += recommend1;
                                                if(recommend2 != null)
                                                    str1 += (", " + recommend2);
                                                if (recommend3 != null)
                                                    str1 += (", " + recommend3);

                                                str1 += "입니다.\n";

                                                String str2 = "감지된 색은 ";
                                                if(tone != null)
                                                    str2 +=  tone + " ";
                                                str2 +=  colors_to_pop  + "입니다.\n";

                                                str2+="추천색은 ";
                                                for(int i=0; i<recomd_string.length; i++){
                                                    str2+=recomd_string[i];
                                                    if(i!=recomd_string.length-1){
                                                        str2+=", ";
                                                    }
                                                }
                                                str2+="입니다.\n";

                                                Intent intent = new Intent(getApplicationContext(), ColorActivity.class);

                                                intent.putExtra("cloth", str1);
                                                intent.putExtra("color",str2);
                                                intent.putExtra("array",recomd_string);
                                                intent.putExtra("title", new String(titles_to_pop));
                                                startActivity(intent);


                                                titles_to_pop = null;
                                            }
                                        })
                                        .setNegativeButton("여자", new DialogInterface.OnClickListener() {
                                            @Override
                                            public void onClick(DialogInterface dialog, int which) {
                                                if (titles_to_pop.equals("블레이저")) {
                                                    recommend1 = "면바지"; //chinos
                                                    recommend2 = "청바지"; //jeans
                                                    recommend3 = "치마";  //skirt
                                                } else if (titles_to_pop.equals("항공점퍼")) {
                                                    recommend1 = "면바지"; //
                                                    recommend2 = "청바지"; //
                                                    recommend3 = "치마";
                                                } else if (titles_to_pop.equals("가디건")) {
                                                    recommend1 = "면바지";
                                                    recommend2 = "청바지";
                                                    recommend3 = "치마";
                                                } else if (titles_to_pop.equals("자켓")) {
                                                    recommend1 = "면바지";
                                                    recommend2 = "청바지";
                                                    recommend3 = "치마";
                                                } else if (titles_to_pop.equals("파카")) {
                                                    recommend1 = "면바지";
                                                    recommend2 = "청바지";
                                                    recommend3 = "트레이닝복 바지";
                                                } else if (titles_to_pop.equals("피코트")) {
                                                    recommend1 = "청바지";
                                                    recommend2 = "면바지";
                                                    recommend3 = "핫팬츠";
                                                } else if (titles_to_pop.equals("코트")) {
                                                    recommend1 = "면바지";
                                                    recommend2 = "청바지";
                                                    recommend3 = "드레스";
                                                } else if (titles_to_pop.equals("케이프")) {
                                                    recommend1 = "치마";
                                                    recommend2 = "면바지";
                                                    recommend3 = "청바지";
                                                }
                                                //상의
                                                else if (titles_to_pop.equals("아노락")) {
                                                    recommend1 = "레깅스";
                                                    recommend2 = "조거";
                                                    recommend3 = "트레이닝복 바지";
                                                } else if (titles_to_pop.equals("블라우스")) {
                                                    recommend1 = "청바지";
                                                    recommend2 = "면바지";
                                                    recommend3 = "치마";
                                                } else if (titles_to_pop.equals("셔츠")) {
                                                    recommend1 = "청바지";
                                                    recommend2 = "조거";
                                                    recommend3 = "면바지";
                                                } else if (titles_to_pop.equals("후드")) {
                                                    recommend1 = "면바지";
                                                    recommend2 = "트레이닝복 바지";
                                                    recommend3 = "핫팬츠";
                                                } else if (titles_to_pop.equals("스웨터")) {
                                                    recommend1 = "청바지";
                                                    recommend2 = "면바지";
                                                    recommend3 = "치마";
                                                } else if (titles_to_pop.equals("탱크탑")) {
                                                    recommend1 = "청바지";
                                                    recommend2 = "치마";
                                                    recommend3 = "핫팬";
                                                } else if (titles_to_pop.equals("티셔츠")) {
                                                    recommend1 = "치마";
                                                    recommend2 = "핫팬츠";
                                                    recommend3 = "조거";
                                                } else if (titles_to_pop.equals("상의")) {
                                                    recommend1 = "청바지";
                                                    recommend2 = "치마";
                                                    recommend3 = "면바지";
                                                } else if (titles_to_pop.equals("홀터넥")) {
                                                    recommend1 = "청바지";
                                                    recommend2 = "치마";
                                                    recommend3 = "핫팬츠";
                                                } else if (titles_to_pop.equals("헨리넥")) {
                                                    recommend1 = "청바지";
                                                    recommend2 = "면바지";
                                                    recommend3 = "핫팬츠";
                                                }
                                                //하의
                                                else if (titles_to_pop.equals("7부바지")) {
                                                    recommend1 = "헨리넥";
                                                    recommend2 = "티셔츠";
                                                    recommend3 = "탱크 탑";
                                                } else if (titles_to_pop.equals("면바지")) {
                                                    recommend1 = "셔츠";
                                                    recommend2 = "티셔츠";
                                                    recommend3 = "스웨터";
                                                } else if (titles_to_pop.equals("핫팬츠")) {
                                                    recommend1 = "탱크 탑";
                                                    recommend2 = "티셔츠";
                                                    recommend3 = "블라우스";
                                                } else if (titles_to_pop.equals("청바지")) {
                                                    recommend1 = "셔츠";
                                                    recommend2 = "티셔츠";
                                                    recommend3 = "블라우스";
                                                } else if (titles_to_pop.equals("조거")) {
                                                    recommend1 = "티셔츠";
                                                    recommend2 = "헨리넥";
                                                    recommend3 = "탱크 탑";
                                                } else if (titles_to_pop.equals("레깅스")) {
                                                    recommend1 = "후드";
                                                    recommend2 = "티셔츠";
                                                    recommend3 = "탱크탑";
                                                } else if (titles_to_pop.equals("반바지")) {
                                                    recommend1 = "티셔츠";
                                                    recommend2 = "상의";
                                                    recommend3 = "탱크 탑";
                                                } else if (titles_to_pop.equals("트레이닝복바지")) {
                                                    recommend1 = "후드";
                                                    recommend2 = "티셔츠";
                                                    recommend3 = "탱크 탑";
                                                } else if (titles_to_pop.equals("치마")) {
                                                    recommend1 = "블라우스";
                                                    recommend2 = "상의";
                                                    recommend3 = "셔츠";
                                                }


//                                                String str1 = "추천옷은 " + recommend1 + ", " + recommend2 + ", " + recommend3 + "입니다.\n";
                                                String str1 = "추천옷은 ";
                                                if(recommend1 != null)
                                                    str1 += recommend1;
                                                if(recommend2 != null)
                                                    str1 += (", " + recommend2);
                                                if (recommend3 != null)
                                                    str1 += (", " + recommend3);

                                                str1 += "입니다.\n";

                                                String str2 = "감지된 색은 ";
                                                if(tone != null)
                                                    str2 +=  tone + " ";
                                                str2 +=  colors_to_pop  + "입니다.\n";

                                                str2+="추천색은 ";
                                                for(int i=0; i<recomd_string.length; i++){
                                                    str2+=recomd_string[i];
                                                    if(i!=recomd_string.length-1){
                                                        str2+=", ";
                                                    }
                                                }
                                                str2+="입니다.\n";

                                                Intent intent = new Intent(getApplicationContext(), ColorActivity.class);

                                                intent.putExtra("cloth", str1);
                                                intent.putExtra("color", str2);
                                                intent.putExtra("array",recomd_string);
                                                intent.putExtra("title", new String(titles_to_pop));

                                                startActivity(intent);

                                                titles_to_pop = null;
                                            }

                                        }).show();

                                handler.sendEmptyMessage(0);

                                col_bit = null;
                                detected_loc = null;
                                recommend1 = null;
                                recommend2 = null;
                                recommend3 = null;
//                                titles_to_pop = null;

                            }

                        });
                    }
                }
                );

                getcol.start();

            }
            else {
                new AlertDialog.Builder(DetectorActivity.this).setTitle("색 추천").setMessage("감지된 옷이 없습니다, 옷 감지를 먼저 해주세요.\n\n")
                        .setNeutralButton("닫기", new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialog, int which) {

                            }
                        })
                        .show();
            }
        }
    }

    public Scalar extract_col_from_detection(Bitmap bit, float[] detLoc) {
        //String tag = "extract_col";
        //Log.i(tag, "start");
        Bitmap b = bit.copy(Bitmap.Config.ARGB_8888, true);
        //Log.i(tag, "img");
        Mat img = new Mat();
        //Log.i(tag, "bitmapTomat");
        Utils.bitmapToMat(b, img);
        Imgproc.resize(img, img, new org.opencv.core.Size(640, 480));
        //Log.i(tag, "cvtColor");
        Imgproc.cvtColor(img, img, Imgproc.COLOR_RGBA2RGB);
//        Imgproc.cvtColor(img, img, Imgproc.COLOR_RGB2HSV);
        //Log.i(tag, "rect");

        // setting bounding box to foreground box
        Rect rect = new Rect((int) detLoc[1], (int) detLoc[0],
                (int) (detLoc[3] - detLoc[1]), (int) (detLoc[2] - detLoc[0]));


        Log.d("rect", "" + detLoc[0] + " " + detLoc[1] + " " + detLoc[2] + " " + detLoc[3] + " " + img.size());
        // setting grabcut parameters
        Mat mask = new Mat();
        mask.setTo(new Scalar(125));
        Mat fgdModel = new Mat();
        Mat bdgModel = new Mat();
        bdgModel.setTo(new Scalar(255, 255, 255));
        fgdModel.setTo(new Scalar(255, 255, 255));
        //Log.i(tag, "before grab");

        //do grabcut
//        Imgproc.grabCut(img, mask, rect, bdgModel, fgdModel, 5, Imgproc.GC_INIT_WITH_RECT);
//        //Log.i(tag, "after grab");
//        Core.compare(mask, new Scalar(Imgproc.GC_PR_FGD), mask, Core.CMP_EQ);
//        Mat foreground = new Mat();
//        foreground.create(img.size(), CV_8UC3);
//        foreground.setTo(new Scalar(255, 255, 255));
//        img.copyTo(foreground, mask);
        //Log.i(tag, "img copy mask");

        Mat foreground = new Mat(img, rect);
        foreground.convertTo(foreground, CvType.CV_32F);
        foreground = foreground.reshape(0, 1);
        Mat labels = new Mat();
        int flags = Core.KMEANS_RANDOM_CENTERS;
        TermCriteria criteria = new TermCriteria(TermCriteria.COUNT, 100, 1);
        int attempts = 10;
        Mat centers = new Mat();
        int k = 5;
        Core.kmeans(foreground, k, labels, criteria, attempts, flags, centers);

        int[] cluster_cnts = new int[k];
        int max = -1, max_idx = -1;
        for (int r = 1; r < labels.rows(); r++) {
            for (int c = 0; c < labels.cols(); c++) {
                //Log.d("hist", ""+r+" "+ c+" "+ Double.toString(bHist.get(r,c)[0]));

                cluster_cnts[(int) labels.get(r, c)[0]]++;
                if (cluster_cnts[(int) labels.get(r, c)[0]] > max) {
                    max = cluster_cnts[(int) labels.get(r, c)[0]];
                    max_idx = (int) labels.get(r, c)[0];
                }
            }
        }

        Log.d("hist", Integer.toString(max_idx));

        for (int r = 1; r < centers.rows(); r++) {
            for (int c = 0; c < centers.cols(); c++) {
                Log.d("hist", "" + r +" " + c + " " + centers.get(r,c)[0]);
        }
    }
        double[] rgb_max = new double[3];
        rgb_max[0] = centers.get(max_idx, 0)[0];
        rgb_max[1] = centers.get(max_idx, 1)[0];
        rgb_max[2] = centers.get(max_idx, 2)[0];



        Log.i("hist", "color: " +": r"  +Double.toString(rgb_max[0]) + " g"+
                Double.toString(rgb_max[1])+" b:"+Double.toString(rgb_max[2]));
        float[] hsv_f = new float[3];
        Color.RGBToHSV((int)rgb_max[0], (int)rgb_max[1], (int)rgb_max[2], hsv_f);

        hsv_f[1] *= 100;
        hsv_f[2] *= 100;

//        hsv_f[0] = b_max; hsv_f[1] = r_max; hsv_f[2] = g_max;
        Scalar ret = new Scalar(hsv_f[0], hsv_f[1], hsv_f[2]);

        Log.d("hist", "ret : "+ret.val[0]+" "+ret.val[1]+" " +ret.val[2]);
        //Scalar tmp = new Scalar(hsv);
        return ret;
    }
}