from flask import Flask, flash, render_template, request, redirect, session, url_for
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField
from flask_wtf import FlaskForm
import cv2 as cv
import subprocess
from moviepy.editor import VideoFileClip
#For Training Imports
from fastai.vision.all import *
from fastai.vision.data import ImageDataLoaders
from path import Path
from matplotlib.axis import Axis
import numpy as np
import matplotlib.pyplot as plt  
from fastai.vision.augment import aug_transforms
from pathlib import Path 
####
import numpy as np
import soundfile as sf 
from IPython.display import Audio
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import librosa.display
####
import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from facenet_pytorch import MTCNN
import cv2
from tqdm import tqdm
import os
import copy
import math
import learn as mylearn


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret key'
Upload_Video = os.path.join('static','videos')
app.config['UPLOAD_VIDEO'] = Upload_Video
Upload_Frames = os.path.join('static','frames')
app.config['UPLOAD_FRAMES'] = Upload_Frames
Upload_AudioEx = os.path.join('static','audioextracted')
app.config['UPLOAD_AUDIOEX'] = Upload_AudioEx
Upload_Audio = os.path.join('static','audios')
app.config['UPLOAD_AUDIO'] = Upload_Audio
Get_spect = os.path.join('static','spectogram')
app.config['GET_SPECT'] = Get_spect


@app.route("/")
def Default():
    return render_template('Home.html')


@app.route("/video")
def video():
    return render_template('VideoUpload.html')

@app.route("/video", methods=['POST'])
def VideoUpload():
    video = request.files['video']
    print(video.filename)
 
    if video:
        filename = secure_filename(video.filename)
        video.save(os.path.join(app.config['UPLOAD_VIDEO'], filename))
        path = os.path.join(app.config['UPLOAD_VIDEO'], filename)

#         cap = cv.VideoCapture(path)

#         i = 0
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             cv.imwrite(os.path.join(app.config['UPLOAD_FRAMES'],f'frame_{i}.jpg'), frame)
#             i += 1

#         cap.release()

        return render_template('ShowVideo.html', filename=filename)

    
@app.route("/predictvideo", methods=['POST'])
def PredictVideo():
    directory_path = app.config['UPLOAD_VIDEO']
    file_list = os.listdir(directory_path)
    video_file = file_list[1]
    print(video_file)
    
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    learn = load_learner('models/video2.pkl', cpu=False)
    def myVideo(file_name):
        mtcnn = MTCNN(margin=20, keep_all=True, post_process=False, device=device)
        # Load video
        v_cap = cv2.VideoCapture(file_name)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Loop through video, taking some no of frames to form a batch   (here, every 30th frame)
        frames = []
        for i in tqdm(range(v_len)):

            # Load frame
            success = v_cap.grab()
            if i % 30 == 0:
                success, frame = v_cap.retrieve()
            else:
                continue
            if not success:
                continue

            # Add to batch
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        #detect faces in frames &  saving frames to file
        f=r"static/frames" + "\\"
        frames_paths = [f+'image' + str(i) +'.jpg' for i in range(len(frames))]
        faces = mtcnn(frames,save_path=frames_paths)
        
    def testing(name):
        #import os
        f=r"static/frames"
        reqd=os.listdir(f)

        if len(reqd)!=0:
            for i in reqd:
                os.remove(f+"\\"+i)

        path=r"static/videos" + "\\" + name
        myVideo(path)
        imgs = get_image_files('static/frames'); imgs[1]
        fnames_test = imgs[:len(imgs)]
        dl = learn.dls.test_dl(fnames_test)
        # dl.show_batch()
        preds = learn.get_preds(dl=dl, with_decoded=True)
        #print(preds[2])

        count_0 = 0
        count_1 = 0

        for element in preds[2]:
            if element == 0:
                count_0 += 1
            elif element == 1:
                count_1 += 1

        if count_0 > count_1:
            return "FAKE"
        elif count_1 > count_0:
            return "REAL"
    print(video_file)
    
    ret = testing(video_file)  
    folder_path = "static/videos/"
        # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Check if the file is a regular file (i.e., not a directory)
        if os.path.isfile(file_path):
            os.remove(file_path)
            
    folder_path = "static/frames/"
        # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Check if the file is a regular file (i.e., not a directory)
        if os.path.isfile(file_path):
            os.remove(file_path)
          
    return ret
        
    
@app.route("/audio")
def audio():
    return render_template('AudioUpload.html')

@app.route("/audio", methods=['POST'])
def AudioUpload():
    audio = request.files['audio']
    print(audio.filename)
    if audio.filename == '':
        return render_template(request.url)
   
    if audio:
        filename = secure_filename(audio.filename)
        audio.save(os.path.join(app.config['UPLOAD_AUDIO'], filename))
        return render_template('ShowAudio.html', filename=filename)


    
@app.route("/predictaudio", methods=['POST'])
def PredictAudio():
    
    directory_path = app.config['UPLOAD_AUDIO']
    file_list = os.listdir(directory_path)
    audio_file = file_list[1]
    print(audio_file)
    y, sr = librosa.load(os.path.join(app.config['UPLOAD_AUDIO'], audio_file))

    # Compute Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

    # Convert to decibels (log scale) for visualization
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Plot Mel spectrogram
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=sr/2)
    # plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')

    # Set output path for spectrogram image
    output_path = app.config['GET_SPECT']

    # Save Mel spectrogram as an image
    plt.savefig('static/spectogram/image.png', bbox_inches='tight')
    
    learn = load_learner('models/spect-trans.pkl', cpu=False)
    imgs = get_image_files('static/spectogram/'); 
    test_img = imgs[0]
    temp = learn.predict(test_img)
    
    folder_path = "static/audios/"

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Check if the file is a regular file (i.e., not a directory)
        if os.path.isfile(file_path):
            os.remove(file_path)

    if temp[0] == "Spoofed":
        return "Fake"
    else:
        return "Real"


@app.route('/display/<filename>')
def display_video(filename):
    return redirect(url_for('static', filename='videos/' + filename), code=301)

@app.route('/display/<filename>')
def display_audio(filename):
    return redirect(url_for('static', filename='audios/' + filename), code=301) 


@app.route('/')
def Back():
    for file in os.listdir(Upload_Frames):
        file_path = os.path.join(app.config['UPLOAD_FRAMES'], file)
        os.remove(file_path)

    return render_template('FileUplaoder.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)