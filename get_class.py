#Master file that will connect all the code files and contain flask code

from flask import Flask, render_template, request, redirect, url_for
# from get_images import get_images, get_path, get_directory
from get_prediction import get_prediction
from generate_html import generate_html
from torchvision import models
import json
from werkzeug.utils import secure_filename
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "./ffmpeg"

# From google colab
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import torch.nn.functional as F
from PIL import Image
import imghdr
import cv2
import numpy as np

import scenedetect
# import pyscenedetect
# from scenedetect.detectors import ContentDetector
# from scenedetect.frame_timecode import FrameTimecode
# from skimage.measure import compare_ssim
from moviepy.video.io.VideoFileClip import VideoFileClip
# import ffmpeg

#App initialization
app = Flask(__name__)
app.config['static'] = 'static/'


# mapping
# imagenet_class_mapping = json.load(open('imagenet_class_index.json'))
class_mapping = json.load(open('class_index.json'))

# # Make sure to pass `pretrained` as `True` to use the pretrained weights:
# model = models.densenet121(pretrained=True)
# # Since we are using our model only for inference, switch to `eval` mode:
# model.eval()

# MY ENSEMBLE MODEL
class ResNet152(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.5): #Added regulatization technique: Dropout - to prevent overfitting due to small dataset size
        super(ResNet152, self).__init__()
        self.resnet = models.resnet152(pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(self.resnet.fc.in_features, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)

class DenseNet169(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet169, self).__init__()
        self.densenet = models.densenet169(pretrained=True)
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)
        
    def forward(self, x):
        return self.densenet(x)


class VGG19(nn.Module):
    def __init__(self, num_classes):
        super(VGG19, self).__init__()
        self.vgg = models.vgg19(pretrained=True)
        # self.vgg.classifier[6] = nn.Linear(self.vgg.classifier[6].in_features, num_classes)
        self.vgg.classifier[-1] = nn.Linear(self.vgg.classifier[-1].in_features, num_classes)
        
    def forward(self, x):
        return self.vgg(x)

# Define the ensemble model
class Ensemble(nn.Module):
    def __init__(self, model_list):
        super(Ensemble, self).__init__()
        self.models = nn.ModuleList(model_list)
        
    def forward(self, x):
        output_list = [model(x) for model in self.models]
        output = torch.stack(output_list, dim=0)
        output = torch.mean(output, dim=0)
        return output
    
num_classes = 2

resnet_model = ResNet152(num_classes)
densenet_model = DenseNet169(num_classes)
vgg_model = VGG19(num_classes)

# resnet_weights = torch.load("models/resnet_model_2.pth", map_location=torch.device('cpu'))
# densenet_weights = torch.load('models/densenet_model_2.pth', map_location=torch.device('cpu'))
# vgg_weights = torch.load('models/VGG_model.pth', map_location=torch.device('cpu'))

# THINGS TO DO: TRAIN MODEL AGAIN AND SAVE STATE_DICT

# Train modified ensemble model, save state_dict and model again. state_dict = model.state_dict()
# Load ensemble model .pth file into ensemble_weights
# Load ensemble model state_dict .pth file into ensemble_model.load_state_dict()

# resnet_model.load_state_dict(resnet_weights['state_dict'])
# densenet_model.load_state_dict(densenet_weights)
# vgg_model.load_state_dict(vgg_weights)

# resnet_model.load_state_dict(resnet_weights['state_dict'])
# densenet_model.load_state_dict(densenet_weights)
# vgg_model.load_state_dict(vgg_weights)

# print(resnet_model.state_dict())

# Do after saving state_dict after training
ensemble_state_dict = torch.load("models/ensemble_model_ac_3_state_dict.pth", map_location = torch.device('cpu'))
ensemble_model = Ensemble([resnet_model, densenet_model, vgg_model])
ensemble_model.load_state_dict(ensemble_state_dict)
ensemble_model.eval()


# def get_image_class(path):
#     get_images(path)
#     path = get_path(path)
#     images_with_tags = get_prediction(model, imagenet_class_mapping, path)
#     print("Printing tags")
#     print(images_with_tags)
#     generate_html(images_with_tags)

def get_image_class(image_file, file_name): #new
    # get_images(path)
    # path = get_path(path)
    path_to_file = "static/" + file_name
    print("PATH:" + path_to_file)
    # images_with_tags = get_prediction(ensemble_model, class_mapping, path_to_file)
    final_cat = get_prediction(ensemble_model, class_mapping, path_to_file, "image")
    # images_with_tags = get_prediction(model, imagenet_class_mapping, path)
    print("Printing tags")
    print(final_cat) 
    return final_cat
    # generate_html(images_with_tags)


# def extractKeyFrames(path_to_file, output_folder, fps=1):
def extract_keyframes(video_path, output_dir): 
    # Using opencv but capturing all the frames
    # cap = cv2.VideoCapture(video_path)
    # # Set the threshold for shot detection
    # threshold = 5000
    # # Set the number of previous frames to use for computing the histogram
    # num_frames_for_hist = 10
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # # Initialize the frame counter and histogram
    # frame_count = 0
    # hist_list = []
    # hist_old = None

    # while cap.isOpened():
    #     # Read a frame from the video file
    #     ret, frame = cap.read()

    #     if not ret:
    #         break

    #     # Increment the frame counter
    #     frame_count += 1

    #     # Convert the frame to grayscale and calculate its histogram
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    #     # If there is a significant change in the histogram (i.e., a shot boundary)
    #     if hist_old is not None and cv2.compareHist(hist_old, hist, cv2.HISTCMP_CHISQR) > threshold:
    #         # Save the keyframe as an image file
    #         output_path = os.path.join(output_dir, f'frame_{frame_count}.jpg')
    #         cv2.imwrite(output_path, frame)

    #     # # Update the previous histogram
    #     hist_old = hist

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize the frame counter and motion detector
    frame_count = 0
    motion_detector = cv2.createBackgroundSubtractorMOG2()

    # Set the minimum motion threshold for keyframe selection
    motion_threshold = 3000

    # Set the minimum time interval between consecutive keyframes (in seconds)
    keyframe_interval = 2

    # Initialize the last keyframe timestamp
    last_keyframe_timestamp = 0

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Total number of frames: {total_frames}')

    while True:
        # Read a frame from the video file
        ret, frame = cap.read()

        if not ret:
            break

        # Increment the frame counter
        frame_count += 1

        # Apply the motion detector to the current frame
        mask = motion_detector.apply(frame)

        # Calculate the total motion in the current frame
        motion = cv2.countNonZero(mask)

        # If the total motion exceeds the threshold and the time interval since the last keyframe has elapsed
        if motion > motion_threshold and frame_count / cap.get(cv2.CAP_PROP_FPS) - last_keyframe_timestamp >= keyframe_interval:
            # Save the current frame as a keyframe
            output_path = os.path.join(output_dir, f'frame_{frame_count}.jpg')
            cv2.imwrite(output_path, frame)

            # Update the last keyframe timestamp
            last_keyframe_timestamp = frame_count / cap.get(cv2.CAP_PROP_FPS)

    # Release the video capture object
    cap.release()
    


def get_video_class(file, file_name):
    name_of_file = os.path.splitext(file_name)[0]
    path_to_file = "static/" + file_name
    print("Video path: " + path_to_file) #Path to entire video
    
    
    output_folder = "static/" + name_of_file

    #Loop to extract key frames from video and make predictions on the key frames and give final
    #prediction based on majority class prediction

    # key_frames_from_video = extractKeyFrames(path_to_file, "outputframes2", 1)
    extract_keyframes(path_to_file, output_folder)

    # print("TYPE OF KEY FRAME: " + str(type(key_frames_from_video[0])))
    # print("Size of key frame array:")
    # print(len(key_frames_from_video))

    #Looping through key frame images to perform prediction
    fire = 0
    nonfire = 0
    no_of_key_frames = 0
    for key_frame in os.listdir(output_folder):
        no_of_key_frames += 1
        image_path = os.path.join(output_folder, key_frame)
        key_frame_cat = get_prediction(ensemble_model, class_mapping, image_path, "video")
        if key_frame_cat == "Fire":
            fire += 1
        else:
            nonfire += 1

    #firevideo3, fire = 10, nf = 25, total = 35
    #nonfirevideo4, fire = 10, nf = 22, total = 32
    threshold_no_of_key_frames = 0.25 * no_of_key_frames
    print("Number of keyframes: " + str(no_of_key_frames))
    print("Threshold number of keyframes: " + str(threshold_no_of_key_frames))
    print("Number of fire: " + str(fire))
    print("Number of non-fire: " + str(nonfire))
    print("Total: " + str(fire+nonfire))
    if fire >= nonfire or fire >= threshold_no_of_key_frames:
        final_cat = "Fire"
    else:
        final_cat = "Non-Fire"

    # final_cat = get_prediction(ensemble_model, class_mapping, path_to_file)
    print("Printing tags")
    print(final_cat)
    return final_cat


#OLD
# @app.route('/')
# def home():
#     return render_template('home.html', final_pred = "", img_path = "")


# @app.route('/', methods=['POST', 'GET'])
# def get_data():
#     if request.method == 'POST':
#         # user = request.form['search'] - old
        
#         file = request.files['uploaded_file'] # new
#         filename = secure_filename(file.filename) #new
#         file.save(os.path.join(app.config['static'], filename)) #new

#         img_path = os.path.join(app.config['static'], filename)
#         final_cat = get_image_class(file, filename) # new
#         return render_template('home.html', final_pred = final_cat, img_path = img_path) #new
        
        # return redirect(url_for('success', name=get_directory(user)))

#NEW
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        file = request.files['uploaded_file']
        if file and allowed_file(file.filename):
            # if file != None:
            #     print("File is not none")
            # else:
            #     print("File is none")
            filename = secure_filename(file.filename)
            print("filename: " + filename)
            file.save(os.path.join(app.config['static'], filename))
            
            
            img_path = os.path.join(app.config['static'], filename)


            # file_type = imghdr.what(img_path)
            # print("File Type: " + str(file_type))
            # if (file_type == None):
            file_extension = os.path.splitext(img_path)[1]
            file_type = file_extension.lower()
            print('File Format: ' + str(file_extension.lower()))
            
            
            
            if file_type in {'.jpeg', '.bmp', '.png', '.gif', '.jpg'}:
                final_cat = get_image_class(file, filename)
                # print("Type of final_cat: " + str(type(final_cat)))
                return render_template('home.html', final_pred=final_cat, file_path = img_path, file_type = 'image')
            elif file_type in {'.mp4', '.avi'}:
                # final_cat = get_video_class(file, filename)
                # return render_template('home.html', final_pred=final_cat, vid_path=img_path)
                print("Video file received: " + filename)
                final_cat = get_video_class(file, filename)
                return render_template('home.html', final_pred=final_cat, file_path = img_path, file_type = 'video')
        else:
            return 'Invalid file type'
    else:
        return render_template('home.html')


# @app.route('/success/<name>')
# def success(name):
#     return render_template('image_class.html')


if __name__ == '__main__' :
    app.run(debug=True)
