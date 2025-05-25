package com.example.screen;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Environment;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;



public class SensorDataLogger implements SensorEventListener {
    private static final String TAG = "SensorDataLogger";
    private FileOutputStream outputStream;
    //临时保存ACCELEROMETER的信息，会实时更新。
    private float[] accelerometerValues = new float[3];
    private float[] linearAccelerometerValues = new float[3];
    //临时保存磁场传感器的信息
    private float[] magneticValues = new float[3];
    private Double[] ThreeDimensionOrientation = new Double[3];
    private final static double radian_to_angle = 180/Math.PI;
    private File file;
    private String action;
    public SensorDataLogger(Context context, String uniqueID, String sensorFileName, String action) {
        File externalAppDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
        this.action = action;
        if (externalAppDir == null) {
            throw new RuntimeException("Couldn't create external files directory");
        }
        file = new File(externalAppDir,  uniqueID + "_" + sensorFileName + "_" + action + "_sensorData.csv");
        try {
            outputStream = new FileOutputStream(file, true); // Append mode
            if (file.length() == 0) {
                writeHeader();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    private void writeHeader() throws IOException {
        String header = "Time,SensorType,X,Y,Z\n";
        outputStream.write(header.getBytes());
        outputStream.flush();
    }
    @Override
    public void onSensorChanged(SensorEvent event) {
        StringBuilder sb = new StringBuilder();
        long nanoTime = System.nanoTime();

        switch (event.sensor.getType()) {
            //Sensor.TYPE_GRAVITY 是 Android 中的一种传感器类型，用于测量设备所受重力的影响。这个传感器返回的三个值代表了设备在 X、Y、Z 轴上的重力加速度分量，单位是米每平方秒（m/s²）。这些值反映了设备相对于地球重力场的方向和强度。
            //
            //X轴的重力加速度 (values[0])：表示设备在 X 轴方向上受到的重力影响。当设备静止时，如果设备平放且屏幕朝上，这个值通常接近于 0。
            //Y轴的重力加速度 (values[1])：表示设备在 Y 轴方向上受到的重力影响。同样地，当设备静止且屏幕朝上时，这个值也接近于 0。
            //Z轴的重力加速度 (values[2])：表示设备在 Z 轴方向上受到的重力影响。当设备静止且屏幕朝上时，这个值通常等于地球的重力加速度，大约是 9.8 m/s²。
            //重力传感器的数据可以用来确定设备相对于地球的方位，例如，当你将设备从水平位置倾斜时，重力传感器的输出会反映出这种变化。这些数据也可以用于各种应用，如游戏控制、设备方向检测、运动检测等。
            case Sensor.TYPE_GRAVITY:
                sb.append(nanoTime).append(",");
                sb.append("Gravity").append(",");
                sb.append(event.values[0]).append(",");//X
                sb.append(event.values[1]).append(",");//Y
                sb.append(event.values[2]).append("\n");//Z
                break;

                //线性加速度计的输出与设备在空间中的移动或加速度变化有关。
            case Sensor.TYPE_ACCELEROMETER:
                //将加速度传感器的信息保存在临时变量中，用于计算三维方向
                //从手机的加速度传感器收集到的三个值分别代表手机在三个不同方向上的加速度信息，具体为X轴、Y轴和Z轴方向上的加速度值。这三个方向在空间坐标系上的含义如下：
                //X轴：通常代表手机宽度的方向，或者是水平方向上的左右移动。当手机在水平面上左右倾斜时，X轴的加速度值会发生变化。
                //Y轴：通常代表手机长度的方向，或者是水平方向上的前后移动。当手机在水平面上前后倾斜时，Y轴的加速度值会发生变化。
                //Z轴：代表垂直于手机屏幕的方向，即手机的厚度方向。由于地球上的任何物体都受到重力加速度的影响，因此当手机处于静止状态时，Z轴的加速度值会接近于重力加速度的值（约9.8m/s²，具体数值取决于手机放置的方式和位置）。当手机上下移动或倾斜时，Z轴的加速度值也会相应变化。
                //需要注意的是，加速度传感器返回的数据值包含了地心引力的影响。例如，当手机平放在桌面上时，Z轴的加速度值会接近于重力加速度G（约9.8m/s²），而在其他方向上（X轴和Y轴），由于手机处于静止状态，加速度值会接近于零（实际上可能由于传感器噪声和误差而略有偏差）。
                //此外，加速度传感器在手机中有多种应用，如步数计算、摇一摇功能、抬手亮屏等。在步数计算中，加速度传感器可以检测用户在行走过程中手机重心在垂直方向上的加速度变化，从而统计步数。在摇一摇功能中，加速度传感器可以检测用户在短时间内快速摇动手机时产生的加速度变化，从而触发相应的应用操作。在抬手亮屏功能中，加速度传感器可以检测用户在拿起手机时产生的加速度变化，从而自动点亮手机屏幕。
                //总的来说，加速度传感器是手机中非常重要的一种传感器，它能够提供手机在不同方向上的加速度信息，为手机的各种应用提供了重要的数据支持。
                accelerometerValues = event.values.clone();
                sb.append(nanoTime).append(",");//Time
                sb.append("Accelerometer").append(",");
                sb.append(event.values[0]).append(",");//X
                sb.append(event.values[1]).append(",");//Y
                sb.append(event.values[2]).append("\n");//Z
                //当加速度传感器发生变化时，计算角度信息
                CalculateOrientation();
                break;
                //Sensor.TYPE_GYROSCOPE 事件触发后，输出的三个值代表了设备围绕三个正交轴（X、Y、Z轴）的角速度。这些值通常以弧度每秒（rad/s）为单位，表示设备在每个轴向上的旋转速度。以下是每个值的具体含义：
                //
                //X轴的角速度 (values[0])：
                //这个值表示设备围绕X轴的旋转速度。在大多数智能手机中，X轴通常被认为是从设备的一侧（左侧）穿过屏幕指向另一侧（右侧）的轴。正值通常表示设备在X轴上逆时针旋转（从设备的顶部向下看），负值表示顺时针旋转。
                //Y轴的角速度 (values[1])：
                //这个值表示设备围绕Y轴的旋转速度。Y轴通常被认为是从设备的底部穿过屏幕指向顶部的轴。正值通常表示设备在Y轴上逆时针旋转（从设备的侧面向前看），负值表示顺时针旋转。
                //Z轴的角速度 (values[2])：
                //这个值表示设备围绕Z轴的旋转速度。Z轴通常被认为是垂直于设备的屏幕，从设备的背面指向正面的轴。正值通常表示设备在Z轴上逆时针旋转（从设备的底部向上看），负值表示顺时针旋转。
                //这些角速度值可以用来计算设备在一段时间内的旋转角度，从而实现各种基于旋转的交互和控制。例如，在游戏或虚拟现实应用中，用户可以通过倾斜和旋转设备来控制视角或游戏角色的动作。
                //
                //在处理陀螺仪数据时，通常需要对数据进行滤波和融合，以减少噪声和误差，提高数据的准确性和稳定性。这可以通过软件算法（如卡尔曼滤波器）或硬件辅助（如传感器融合算法）来实现。
            case Sensor.TYPE_GYROSCOPE:
                sb.append(nanoTime).append(",");//Time
                sb.append("Gyroscope").append(",");
                sb.append(event.values[0]*radian_to_angle).append(",");//X
                sb.append(event.values[1]*radian_to_angle).append(",");//Y
                sb.append(event.values[2]*radian_to_angle).append("\n");//Z
                break;

            case Sensor.TYPE_LINEAR_ACCELERATION:
                linearAccelerometerValues = event.values.clone();
                sb.append(nanoTime).append(",");
                sb.append("Magnetic Field").append(",");
                sb.append(event.values[0]).append(",");//X
                sb.append(event.values[1]).append(",");//Y
                sb.append(event.values[2]).append("\n");//Z
                break;
                //Sensor.TYPE_MAGNETIC_FIELD 是 Android 中用于获取磁场传感器数据的传感器类型。磁场传感器，也称为磁力计，
                // 用于测量设备周围环境的磁场。它返回的三个值分别代表设备在 X、Y、Z 轴上的磁场强度，单位是微特斯拉（uT）。
            case Sensor.TYPE_MAGNETIC_FIELD:
                magneticValues = event.values.clone();
                sb.append(nanoTime).append(",");
                sb.append("Magnetic Field").append(",");
                sb.append(event.values[0]).append(",");//X
                sb.append(event.values[1]).append(",");//Y
                sb.append(event.values[2]).append("\n");//Z
                //当磁场传感器发生变化时，计算角度信息。
                CalculateOrientation();
                break;

                //sensor.TYPE_PROXIMITY传感器通常被称为接近传感器或距离传感器。它的主要作用是检测手机与用户脸部的距离，以便在通话时自动关闭屏幕，防止误触。接近传感器通常位于手机的听筒附近。
                //
                //接近传感器返回的三个值通常包括：
                //
                //距离值 (values[0])：表示传感器检测到的距离，通常以厘米为单位。当物体靠近传感器时，这个值会减小。如果物体距离传感器非常近（通常小于传感器的最大检测距离），这个值可能会接近0。
                //光强度 (values[1])：某些接近传感器集成了光线传感器的功能，可以提供环境光强度的信息。
                //接近状态 (values[2])：表示物体是否靠近传感器，通常是一个布尔值，接近时为1，远离时为0。
            case Sensor.TYPE_PROXIMITY:
                sb.append(nanoTime).append(",");//Time
                sb.append("Proximity").append(",");
                sb.append(event.values[0]).append(",");
                sb.append(event.values[0]).append(",");
                sb.append(event.values[0]).append("\n");
                break;
        }
        // 写入文件
        try {
            outputStream.write(sb.toString().getBytes());
            outputStream.flush();
        } catch (IOException e) {
            Log.e(TAG, "Error writing to file", e);
        }
    }
    //获得设备相对于地球坐标系的三个角度。
    private void CalculateOrientation() {
        float[] rotationMatrix = new float[9];
        float[] orientationAngles = new float[3];

        //使用 SensorManager.getRotationMatrix 方法来计算旋转矩阵。
        //一旦你有了旋转矩阵，你可以使用 SensorManager.getOrientation 方法来获取设备的方位角、俯仰角和翻滚角
        //使用 SensorManager.getOrientation 方法可以获得设备相对于地球坐标系的三个角度，
        //分别是方位角（Azimuth）、俯仰角（Pitch）和翻滚角（Roll）。这些角度描述了设备在三维空间中的旋转状态。以下是每个角度的具体含义。
        //SensorManager.getOrientation 方法返回的角度值是以弧度为单位的，你可能需要将它们转换为角度。
        //方位角（Azimuth）：
        //方位角描述的是设备顶部相对于地球磁场北的方向的角度。它表示设备在水平面上的旋转。
        //当设备正面指向地球的磁北极时，方位角通常被定义为 0 度。
        //当设备正面从指向磁北极转向指向地球的磁南极时，方位角从 0 度增加到 180 度。
        //当设备正面从指向磁北极转向指向地球的磁南极时，方位角从 0 度增加到 -180 度（或 360 度）。
        //俯仰角（Pitch）：
        //俯仰角描述的是设备相对于水平面的倾斜程度，它表示设备围绕 x 轴的旋转。
        //当设备平放且屏幕朝上时，俯仰角为 0 度。
        //当设备前端（屏幕）向上抬起时，俯仰角为正值，表示向上倾斜。
        //当设备后端（底部）向上抬起时，俯仰角为负值，表示向下倾斜。
        //翻滚角（Roll）：
        //翻滚角描述的是设备相对于垂直于地面的平面的旋转，它表示设备围绕 y 轴的旋转。
        //当设备平放且屏幕朝上时，翻滚角为 0 度。
        //当设备向左倾斜时，翻滚角为正值，表示向左旋转。
        //当设备向右倾斜时，翻滚角为负值，表示向右旋转。
        SensorManager.getRotationMatrix(rotationMatrix, null, accelerometerValues, magneticValues);
        SensorManager.getOrientation(rotationMatrix, orientationAngles);
        // 将弧度转换为角度
        ThreeDimensionOrientation[0] = Math.toDegrees(orientationAngles[0]); // 方位角
        ThreeDimensionOrientation[1] = Math.toDegrees(orientationAngles[1]); // 俯仰角
        ThreeDimensionOrientation[2] = Math.toDegrees(orientationAngles[2]); // 翻滚角
        StringBuilder sb = new StringBuilder();
        long nanoTime = System.nanoTime();
        sb.append(nanoTime).append(",");//Time
        sb.append("OrientationAngles").append(",");
        sb.append(ThreeDimensionOrientation[0]).append(",");//X
        sb.append(ThreeDimensionOrientation[1]).append(",");//Y
        sb.append(ThreeDimensionOrientation[2]).append("\n");//Z
        //将角度信息写入文件
        try {
            outputStream.write(sb.toString().getBytes());
            outputStream.flush();
        } catch (IOException e) {
            Log.e(TAG, "Error writing to file", e);
        }
        return;
        // 使用这些角度值来更新 UI 或执行其他操作
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // 处理精度变化
    }
    public void close() {
        try {
            if (outputStream != null) {
                outputStream.close();
            }
        } catch (IOException e) {
            Log.e(TAG, "Error closing file", e);
        }
    }
}
