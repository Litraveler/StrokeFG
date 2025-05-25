package com.example.screen;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.View;
import android.widget.EditText;
import android.widget.RadioGroup;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

public class SelectionInterface extends AppCompatActivity{
    private EditText editText;
    private String uniqueID;
    private String action = "sit";
    private RadioGroup actionRatioGroup;
    private static final int PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE = 112;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_choose);
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE);
            System.out.println("成功");
        } else {
            // 已经获得权限，可以进行文件操作
        }
        editText = findViewById(R.id.editText);
        editText.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence charSequence, int i, int i1, int i2) {

            }

            @Override
            public void onTextChanged(CharSequence charSequence, int i, int i1, int i2) {
                uniqueID = charSequence.toString();
            }

            @Override
            public void afterTextChanged(Editable editable) {
                uniqueID = editable.toString();
            }
        });
        actionRatioGroup = findViewById(R.id.action);
        actionRatioGroup .setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                // 根据选中的RadioButton的ID执行操作
                switch (checkedId) {
                    case R.id.sit:
                        // 坐着
                        action = "sit";
                        break;
                    case R.id.walk:
                        // 行走
                        action = "walk";
                        break;
                }
            }
        });
    }

    public void onTouchGestureButtonClick(View view) {
        Intent intent = new Intent(this, TouchGestureActivity.class);
        intent.putExtra("uniqueID", uniqueID);
        intent.putExtra("action", action);
        startActivity(intent);
    }


}
