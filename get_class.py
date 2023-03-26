#Master file that will connect all the code files and contain flask code

from flask import Flask, render_template, request, redirect, url_for
from get_images import get_images, get_path, get_directory
from get_prediction import get_prediction
from generate_html import generate_html
from torchvision import models
import json
from werkzeug.utils import secure_filename
import os

# From google colab
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import torch.nn.functional as F
from PIL import Image

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
    final_cat = get_prediction(ensemble_model, class_mapping, path_to_file)
    # images_with_tags = get_prediction(model, imagenet_class_mapping, path)
    print("Printing tags")
    print(final_cat) 
    return final_cat
    # generate_html(images_with_tags)


@app.route('/')
def home():
    return render_template('home.html', final_pred = "", img_path = "")


@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        # user = request.form['search'] - old
        
        file = request.files['uploaded_file'] # new
        filename = secure_filename(file.filename) #new
        file.save(os.path.join(app.config['static'], filename)) #new

        img_path = os.path.join(app.config['static'], filename)
        final_cat = get_image_class(file, filename) # new
        return render_template('home.html', final_pred = final_cat, img_path = img_path) #new
        
        # return redirect(url_for('success', name=get_directory(user)))


@app.route('/success/<name>')
def success(name):
    return render_template('image_class.html')


if __name__ == '__main__' :
    app.run(debug=True)
