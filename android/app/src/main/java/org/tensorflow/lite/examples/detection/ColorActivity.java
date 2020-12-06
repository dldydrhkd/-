package org.tensorflow.lite.examples.detection;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Color;
import android.graphics.PorterDuff;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;

import java.util.Arrays;
import java.util.HashSet;

public class ColorActivity extends AppCompatActivity {

    final static HashSet<String> top = new HashSet(Arrays.asList("아노락", "블레이저", "블라우스",
            "항공점퍼", "셔츠", "가디건", "홀터넥", "헨리넥", "후드", "자켓", "저지", "파카", "피코트", "스웨터", "탱크탑",
            "티셔츠", "상의", "터틀넥", "코트", "원피스", "점프슈트", "잠옷"));

    final static HashSet<String> bottom = new HashSet<>(Arrays.asList("7부바지", "면바지","팬티", "핫팬츠",
            "청바지", "조거", "레깅스", "반바지", "치마", "트레이닝복바지","트렁크"));

    private ImageView[] ImageArray = new ImageView[5];
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_color);
        Intent intent = getIntent();

        TextView tx1 = (TextView)findViewById((R.id.cloth_recommend_text));
        TextView tx2 = (TextView)findViewById((R.id.color_recommend_text));
        String title = intent.getExtras().getString("title");

        Log.d("get_title", title);
        if(bottom.contains(title)) {
            Log.d("get_title", "bottom");
            ImageArray[0] = (ImageView) findViewById(R.id.top_imageView1);
            ImageArray[1] = (ImageView) findViewById(R.id.top_imageView2);
            ImageArray[2] = (ImageView) findViewById(R.id.top_imageView3);
            ImageArray[3] = (ImageView) findViewById(R.id.top_imageView4);
            ImageArray[4] = (ImageView) findViewById(R.id.top_imageView5);
        }
        else if(top.contains(title)){
            Log.d("get_title", "top");
            ImageArray[0] = (ImageView) findViewById(R.id.bot_imageView1);
            ImageArray[1] = (ImageView) findViewById(R.id.bot_imageView2);
            ImageArray[2] = (ImageView) findViewById(R.id.bot_imageView3);
            ImageArray[3] = (ImageView) findViewById(R.id.bot_imageView4);
            ImageArray[4] = (ImageView) findViewById(R.id.bot_imageView5);
        }

        String cloth = intent.getExtras().getString("cloth");
        String color = intent.getExtras().getString("color");
        String array[] = intent.getExtras().getStringArray("array");


        for(int i=0; i<array.length; i++){
            ImageArray[i].setVisibility(View.VISIBLE);
            ImageArray[i].setContentDescription(array[i]);
            Log.d("out_col", array[i].toString());
            ImageArray[i].setColorFilter(Color.parseColor(find_color(array[i])), PorterDuff.Mode.SRC_IN);
        }
        tx1.setText(cloth);
        tx2.setText(color);
    }

    public String find_color(String s){
        if(s.equals("회색")){
            return "#A9A9A9";
        }
        else if(s.equals("검은색")){
            return "#130C0E";
        }
        else if(s.equals("하늘색")){
            return "#F0FFFF";
        }
        else if(s.equals("노란색")){
            return "#FFD400";
        }
        else if(s.equals("와인색")){
            return "#760C0C";
        }
        else if(s.equals("베이지")){
            return "#F5F5DC";
        }
        else if(s.equals("흰색")){
            return "#FFFFFB";
        }
        else if(s.equals("카키")){
            return "#F0E68C";
        }
        else if(s.equals("파란색")){
            return "#0000FF";
        }
        else if(s.equals("갈색")){
            return "#A52A2A";
        }
        else if(s.equals("남색")){
            return "#000080";
        }
        else if(s.equals("빨간색")){
            return "#FF0000";
        }
        else{
            return "#000000";
        }
    }
}