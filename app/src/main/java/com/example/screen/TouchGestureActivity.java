package com.example.screen;

import android.content.Context;
import android.content.Intent;
import android.hardware.Sensor;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.View;
import android.widget.ImageView;
import android.widget.ProgressBar;

import androidx.appcompat.app.AppCompatActivity;

public class TouchGestureActivity extends AppCompatActivity {
    private String uniqueID;
    private String action;
    private TouchEventLogger touchEventLogger;
    private SensorDataLogger sensorDataLogger;
    private SensorManager sensorManager;
    private String touchType = "NULL";
    private static String type = "Draw";
    private String flag = "0";
    private static String fileName = "touch_gesture";
    private int count = 0;
    // 图片资源数组
    private static final int[] GESTURE_DRAWABLES = {
            R.drawable.gesture_1, R.drawable.gesture_2, R.drawable.gesture_3, R.drawable.gesture_4, R.drawable.gesture_5,
            R.drawable.gesture_6, R.drawable.gesture_7, R.drawable.gesture_8, R.drawable.gesture_9, R.drawable.gesture_10,
            R.drawable.gesture_11, R.drawable.gesture_12, R.drawable.gesture_13, R.drawable.gesture_14, R.drawable.gesture_15,
            R.drawable.gesture_16, R.drawable.gesture_17, R.drawable.gesture_18, R.drawable.gesture_19, R.drawable.gesture_20,
            R.drawable.gesture_21, R.drawable.gesture_22, R.drawable.gesture_23, R.drawable.gesture_24, R.drawable.gesture_25,
            R.drawable.gesture_26, R.drawable.gesture_27, R.drawable.gesture_28, R.drawable.gesture_29, R.drawable.gesture_30,
            R.drawable.success
    };
    private int currentImageIndex = 0;
    private int currentInputCount = 0;
    private ProgressBar progressTotal;
    private ProgressBar progressCurrent;
    private ImageView ivCurrentHint;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_touchgesture);
        progressTotal = findViewById(R.id.progressTotal);
        progressCurrent = findViewById(R.id.progressBar);
        progressTotal.setMax(GESTURE_DRAWABLES.length);
        progressCurrent.setMax(7);
        ivCurrentHint = findViewById(R.id.ivCurrentHint);
        updateCurrentImage();
        // 初始化 DrawView
        DrawView drawView = findViewById(R.id.drawView);
        //获取传递的信息
        Intent intent = getIntent();
        action = intent.getStringExtra("action");
        uniqueID = intent.getStringExtra("uniqueID");
        //记录触摸信息
        touchEventLogger = new TouchEventLogger(uniqueID, fileName, action);
        //记录传感器信息
        sensorDataLogger = new SensorDataLogger(this, uniqueID, fileName, action);
        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        sensorManager.registerListener(sensorDataLogger, sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY), SensorManager.SENSOR_DELAY_NORMAL);
        sensorManager.registerListener(sensorDataLogger, sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SensorManager.SENSOR_DELAY_NORMAL);
        sensorManager.registerListener(sensorDataLogger, sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE), SensorManager.SENSOR_DELAY_NORMAL);
        sensorManager.registerListener(sensorDataLogger, sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD), SensorManager.SENSOR_DELAY_NORMAL);
        sensorManager.registerListener(sensorDataLogger, sensorManager.getDefaultSensor(Sensor.TYPE_PROXIMITY), SensorManager.SENSOR_DELAY_NORMAL);
    }
    private void updateCurrentImage() {
        if (currentImageIndex < GESTURE_DRAWABLES.length) {
            ivCurrentHint.setImageResource(GESTURE_DRAWABLES[currentImageIndex]);
            ivCurrentHint.setTag(String.valueOf(currentImageIndex + 1)); // tag从1开始
            progressTotal.setProgress(currentImageIndex);
        }
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (currentImageIndex >= GESTURE_DRAWABLES.length) {
            return false; // 所有图片已完成
        }

        String temTouchType = "NULL";
        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                temTouchType = "Down_" + ivCurrentHint.getTag();
                break;
            case MotionEvent.ACTION_MOVE:
                temTouchType = "Move_" + ivCurrentHint.getTag();
                break;
            case MotionEvent.ACTION_UP:
                temTouchType = "Up_" + ivCurrentHint.getTag();
                handleInputCompletion();
                break;
        }

        touchEventLogger.saveTouchEvent(event, temTouchType);
        return true;
    }
    private void handleInputCompletion() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                currentInputCount++;
                progressCurrent.setProgress(currentInputCount);

                if (currentInputCount >= 7) {
                    currentImageIndex++;
                    currentInputCount = 0;
                    progressCurrent.setProgress(0);

                    if (currentImageIndex < GESTURE_DRAWABLES.length) {
                        updateCurrentImage();
                    } else {
                        // 所有图片完成处理
                        ivCurrentHint.setVisibility(View.GONE);
                        progressCurrent.setVisibility(View.GONE);
                    }
                }
            }
        });
    }
}
