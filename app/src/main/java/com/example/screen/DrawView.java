package com.example.screen;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

public class DrawView extends View {
    private Paint drawPaint;
    private Path drawPath;
    private float touchX, touchY;

    public DrawView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        drawPaint = new Paint();
        drawPaint.setColor(Color.GREEN); // 设置触摸点的颜色
        drawPaint.setStrokeWidth(20); // 设置触摸点的宽度
        drawPaint.setStyle(Paint.Style.STROKE);
        drawPath = new Path();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        canvas.drawPath(drawPath, drawPaint); // 绘制路径
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        touchX = event.getX();
        touchY = event.getY();

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                drawPath.moveTo(touchX, touchY); // 开始绘制
                break;
            case MotionEvent.ACTION_MOVE:
                drawPath.lineTo(touchX, touchY); // 绘制线条
                break;
            case MotionEvent.ACTION_UP:
                drawPath.reset();
                break;
        }
        invalidate(); // 刷新视图
        // 手动调用父 Activity 的 onTouchEvent 方法
        if (getContext() instanceof TouchGestureActivity) {
            ((TouchGestureActivity) getContext()).onTouchEvent(event);
        }
        return true; // 返回 true，表示事件已被处理
    }
}
