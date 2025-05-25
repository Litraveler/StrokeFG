    package com.example.screen;
    import android.os.Environment;
    import android.util.Log;
    import android.view.MotionEvent;

    import java.io.File;
    import java.io.FileOutputStream;
    import java.io.IOException;

    public class TouchEventLogger {
        private static final String TAG = "TouchEventLogger";
        private TouchEventLogger sInstance;
        private FileOutputStream outputStream;
        private File file;
        private int count;
        private String action;

        public TouchEventLogger(String uniqueID, String sensorFileName, String action) {
            File externalAppDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
            this.action = action;
            if (externalAppDir == null) {
                throw new RuntimeException("Couldn't create external files directory");
            }
            file = new File(externalAppDir, uniqueID + "_" + sensorFileName + "_" + action + "_touchData.csv");
    //        System.out.println("创建成功1"+file.getAbsolutePath());
            try {
                outputStream = new FileOutputStream(file, true); // Append mode
    //            System.out.println("创建成功2");
                if (file.length() == 0) {
                    writeHeader();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        private void writeHeader() throws IOException {
            String header = "ACTION_TYPE,Time,X,Y,SizeMajor,SizeMinor,Orientation,Pressure,Size\n";
            outputStream.write(header.getBytes());
            outputStream.flush();
        }


        public void saveTouchEvent(MotionEvent event, String ACTION_TYPE) {
            StringBuilder sb = new StringBuilder();

            try {
                long nanoTime = System.nanoTime();
                System.out.println("数量：" + event.getPointerCount());
                for (int i = 0; i < event.getPointerCount(); i++) {
                    int tempIndex = event.findPointerIndex(i);
                    if (tempIndex != -1){
                        //测试正确
                        Log.d("TouchEventLogger:", String.valueOf(count));
                        count++;
                        sb.append(ACTION_TYPE).append(",");
                        //时间戳
                        sb.append(nanoTime).append(",");
                        //X坐标
                        sb.append(event.getX(tempIndex)).append(",");
                        //Y坐标
                        sb.append(event.getY(tempIndex)).append(",");
                        //SizeMajor: 触摸点主要轴的长度，表示触摸点的最大直径。
                        sb.append(event.getAxisValue(MotionEvent.AXIS_TOUCH_MAJOR)).append(",");
                        //SizeMinor触摸点次要轴的长度，表示触摸点的最小直径。
                        sb.append(event.getAxisValue(MotionEvent.AXIS_TOUCH_MINOR)).append(",");
                        //Orientation：触摸点的方向，通常用于描述笔的倾斜角度。。
                        sb.append(event.getAxisValue(MotionEvent.AXIS_ORIENTATION)).append(",");
                        //压力pressure sb.append("Pressure: ").append(event.getAxisValue(MotionEvent.AXIS_PRESSURE,i)).append("&");
                        sb.append(event.getPressure(tempIndex)).append(",");
                        //触摸区域Area sb.append("Touch Area: ").append(event.getAxisValue(MotionEvent.AXIS_SIZE,i)).append("\n");
                        sb.append(event.getSize(tempIndex)).append("\n");
                    }
                }
                outputStream.write(sb.toString().getBytes());
                outputStream.flush();
            } catch (IOException e) {
                Log.e(TAG, "Error writing touch event to file", e);
            }
        }
        public void close() {
            try {
                if (outputStream != null) {
                    outputStream.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
