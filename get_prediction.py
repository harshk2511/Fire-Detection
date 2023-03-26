# Code to make transformation on the image and use the model to make predictions
# Change model - Load my model

import json
import pandas as pd
import io
import glob
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

# def get_category(model, imagenet_class_mapping, image_path):
#     print("image path", image_path)
#     with open(image_path, 'rb') as file:
#         image_bytes = file.read()
#         # print("image bytes")
#         # print(image_bytes)
#     transformed_image = transform_image(image_bytes=image_bytes)
#     outputs = model.forward(transformed_image)
    
#     _, category = outputs.max(1)
    
#     predicted_idx = str(category.item())
#     return imagenet_class_mapping[predicted_idx]

def get_category(model, class_mapping, image_path):
    print("image path", image_path)
    with open(image_path, 'rb') as file:
        image_bytes = file.read()
        # print("image bytes")
        # print(image_bytes)
    transformed_image = transform_image(image_bytes=image_bytes)
    
    with torch.no_grad():
        output = model(transformed_image)
    
    #outputs = model.forward(transformed_image) #old
    
    _, predicted = torch.max(output.data, 1)
    #_, category = outputs.max(1) #old

    # Print the predicted class
    confidence_val = _.item()
    print("Category: " + str(predicted.item()))
    print("Confidence: " + str(confidence_val))
    print("Outputs: ")
    print(output)
    fire_conf = abs(output[0][0].item())
    non_fire_conf = abs(output[0][1].item())
    print(fire_conf)
    print(non_fire_conf)


    # if (confidence_val >= 1.5):
    #     print("Predicted class: Fire")
    #     return "Fire"
    # else:
    #     print("Predicted class: Non-Fire")
    #     return "Non-fire"

    diff_f_nf = abs(fire_conf - non_fire_conf)
    final_conf_val = max(fire_conf, non_fire_conf)
    if (diff_f_nf >= 0.049):
        if (final_conf_val == fire_conf):
            final_cat = "Non-Fire"
            print("2. FINAL CATEGORY: NON_FIRE")
        else:
            final_cat = "Fire"
            print("2. FINAL CATEGORY: FIRE")
    else:
    # final_conf_val = max(fire_conf, non_fire_conf) 
        if (final_conf_val == fire_conf):
            final_cat = "Fire"
            print("FINAL CATEGORY: FIRE")
        else:
            final_cat = "Non-Fire"
            print("FINAL CATEGORY: NON_FIRE")
    
    return final_cat
    # predicted_idx = str(category.item())
    # return class_mapping[predicted_idx]

# def get_prediction(model, imagenet_class_mapping, path_to_directory):   #OLD
#     path_to_directory = r"static\URL_https___brightthemag.com_can-a-dog-at-school-help-struggling-kids-7317a8eff20d"
#     files = glob.glob(path_to_directory+'/*')

#     print(path_to_directory, files)
#     image_with_tags = {}
#     for image_file in files:
#         image_with_tags[image_file] = get_category(model, imagenet_class_mapping, image_path=image_file)[1]
#     print("get prediction.py ouput")
#     print(image_with_tags)
#     return image_with_tags

def get_prediction(model, class_mapping, path_to_file):
    #path_to_directory = r"static\URL_https___brightthemag.com_can-a-dog-at-school-help-struggling-kids-7317a8eff20d"
    #files = glob.glob(path_to_directory+'/*')

    #print(path_to_directory, files)
    # dict_of_image_to_category = {}
    # dict_of_image_to_category[path_to_file] = get_category(model, class_mapping, image_path=path_to_file)
    final_cat = get_category(model, class_mapping, image_path=path_to_file)
    
    # image_with_tags = {}
    # for image_file in files:
    #     image_with_tags[image_file] = get_category(model, imagenet_class_mapping, image_path=image_file)[1]
    print("get prediction.py ouput")
    # print(image_with_tags)
    # return image_with_tags
    # print(dict_of_image_to_category)
    
    # return dict_of_image_to_category
    return final_cat

