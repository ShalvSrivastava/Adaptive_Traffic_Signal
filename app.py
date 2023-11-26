#!/usr/bin/env python
# coding: utf-8

# In[1]:


# app.py
# app.py
# app.py
from traffic_video import process_video
from flask import Flask, render_template, request, send_file
from traffic_analysis import analyze_traffic, calculate_green_light_time
import os
import base64
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'user_input.jpg')
        uploaded_file.save(image_path)

        density, vehicle_count,result_image = analyze_traffic(image_path)
        green_light_time = calculate_green_light_time(density)

        return render_template('result.html', result_image_base64=result_image,
                               green_light_time=green_light_time, vehicle_count=vehicle_count,density=density)
    

    return render_template('index.html')

@app.route('/image/<filename>')
def get_image(filename):
    return send_file(f'static/uploads/{filename}', mimetype='image/jpeg')


@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    if request.method == 'POST':
        uploaded_video = request.files['videoFile']
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'user_input_video.mp4')
        uploaded_video.save(video_path)

        total_cars, car_count = process_video(video_path)

        return render_template('result_video.html', total_cars=total_cars, car_count=car_count)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)




# In[ ]:




