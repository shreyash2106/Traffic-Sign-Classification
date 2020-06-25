import os
from flask import Flask, flash, render_template, redirect, request, send_file, url_for
from keras.models import load_model
from numpy import squeeze
import random
from werkzeug.utils import secure_filename
from imageio import imread
from PIL import Image

EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpeg', 'gif'])
CHARACTERS = list(set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))

def is_valid_file_type(filename):
    allowed = filename.rsplit('.', 1)[1].lower() in EXTENSIONS
    return '.' in filename and allowed

def generate_random_name(filename):
    """Generates a random 3 character name for the image file.  """
    ext = filename.split('.')[-1]
    rand = [random.randint(0, len(CHARACTERS) - 1) for _ in range(3)]
    chars = ''.join([CHARACTERS[i] for i in rand])
    new_name = "{new_fn}.{ext}".format(new_fn=chars, ext=ext)
    new_name = secure_filename(new_name)
    return new_name

def load_image(filepath):
    """ Modifies image data so it is ready for prediction according ot model standards. """
    image_data = imread(filepath)[:, :, :3]
    image_data = image_data / 255.
    image_data = image_data.reshape((-1, 30, 30, 3))
    return image_data

def resize_image(filepath):
    """ Converts input image to 30px by 30px thumbnail if not that size
    and save it back to the source file """
    img = Image.open(filepath)
    thumb = None
    w, h = img.size

    # if it is exactly 128x128, do nothing
    if w == 30 and h == 30:
        return True

    # if the width and height are equal, scale down
    if w == h:
        thumb = img.resize((30, 30), Image.BICUBIC)
        thumb.save(filepath)
        return True

    # when the image's width is smaller than the height
    if w < h:
        # scale so that the width is 128px
        ratio = w / 30.
        w_new, h_new = 30, int(h / ratio)
        thumb = img.resize((w_new, h_new), Image.BICUBIC)

        # crop the excess
        top, bottom = 0, 0
        margin = h_new - 30
        top, bottom = margin // 2, 30 + margin // 2
        box = (0, top, 30, bottom)
        cropped = thumb.crop(box)
        cropped.save(filepath)
        return True

    # when the image's height is smaller than the width
    if h < w:
        # scale so that the height is 128px
        ratio = h / 30.
        w_new, h_new = int(w / ratio), 30
        thumb = img.resize((w_new, h_new), Image.BICUBIC)

        # crop the excess
        left, right = 0, 0
        margin = w_new - 30
        left, right = margin // 2, 30 + margin // 2
        box = (left, 0, right, 30)
        cropped = thumb.crop(box)
        cropped.save(filepath)
        return True
    return False

#dictionary to label all traffic signs class.
classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)',      
            3:'Speed limit (50km/h)',       
            4:'Speed limit (60km/h)',      
            5:'Speed limit (70km/h)',    
            6:'Speed limit (80km/h)',      
            7:'End of speed limit (80km/h)',     
            8:'Speed limit (100km/h)',    
            9:'Speed limit (120km/h)',     
           10:'No passing',   
           11:'No passing veh over 3.5 tons',     
           12:'Right-of-way at intersection',     
           13:'Priority road',    
           14:'Yield',     
           15:'Stop',       
           16:'No vehicles',       
           17:'Veh > 3.5 tons prohibited',       
           18:'No entry',       
           19:'General caution',     
           20:'Dangerous curve left',      
           21:'Dangerous curve right',   
           22:'Double curve',      
           23:'Bumpy road',     
           24:'Slippery road',       
           25:'Road narrows on the right',  
           26:'Road work',    
           27:'Traffic signals',      
           28:'Pedestrians',     
           29:'Children crossing',     
           30:'Bicycles crossing',       
           31:'Beware of ice/snow',
           32:'Wild animals crossing',      
           33:'End speed + passing limits',      
           34:'Turn right ahead',     
           35:'Turn left ahead',       
           36:'Ahead only',      
           37:'Go straight or right',      
           38:'Go straight or left',      
           39:'Keep right',     
           40:'Keep left',      
           41:'Roundabout mandatory',     
           42:'End of no passing',      
           43:'End no passing veh > 3.5 tons' }