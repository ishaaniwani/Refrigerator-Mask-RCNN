import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import firebase_admin
from firebase_admin import credentials, firestore, storage
import json
from PIL import Image, ExifTags
from google.cloud import storage as storage2
import uuid
import datetime

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR) # To find the local version of the library
from mrcnn import utils
import mrcnn.model as modellib
import visualize2

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco")) # To find local version
import coco

# %matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
	utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
datetime = ""
cred = credentials.Certificate('/home/ishaan/foodtrack_cred.json')
firebase_admin.initialize_app(cred, {'storageBucket': 'foodtrack-4a83e.appspot.com'})

class InferenceConfig(coco.CocoConfig):
	# Set batch size to 1 since we'll be running inference on
	# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['null', 'null', 'null', 'null', 'null', 'null',
                   'null', 'null', 'null', 'null', 'null',
                   'null', 'null', 'null', 'null', 'null',
                   'null', 'null', 'null', 'null', 'null', 'null', 'null',
                   'null', 'null', 'null', 'null', 'null', 'null',
                   'null', 'null', 'null', 'null', 'null',
                   'null', 'null', 'null', 'null',
                   'null', 'null', 'null', 'wine glass', 'ignore',
                   'fork', 'knife', 'spoon', 'ignore', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'null',
                   'ignore', 'null', 'null', 'null', 'null', 'null',
                   'null', 'null', 'null', 'ignore', 'null',
                   'null', 'ignore', 'null', 'null', 'null', 'null',
                   'null', 'null', 'null']

class_names2 = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

class_names3 = ['object', 'object', 'object', 'object', 'object', 'object',
                   'object', 'object', 'object', 'object', 'object',
                   'object', 'object', 'object', 'object', 'object',
                   'object', 'object', 'object', 'object', 'object', 'object', 'object',
                   'object', 'object', 'object', 'object', 'object', 'object',
                   'object', 'object', 'object', 'object', 'object',
                   'object', 'object', 'object', 'object',
                   'object', 'object', 'object', 'object', 'ignore',
                   'object', 'object', 'object', 'ignore', 'object', 'object',
                   'object', 'object', 'object', 'object', 'object', 'object',
                   'object', 'object', 'object', 'object', 'object', 'object',
                   'ignore', 'object', 'object', 'object', 'object', 'object',
                   'object', 'object', 'object', 'object', 'object',
                   'object', 'object', 'object', 'object', 'object', 'object',
                   'object', 'object', 'object']

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage2.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

while True:
    print("starts here")
    # loop this
    bucket_name = "010_refridgenators"
    filename = "initial.jpg"
    newfilename = uuid.uuid4()

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ishaan/Downloads/google_cred.json"
    #download_blob(bucket_name, filename, filename)

    img = Image.open(filename)
    exif = { ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS }
    if datetime == exif['DateTime']:
        print('stop right there criminal')
        continue
    datetime = exif['DateTime']

    image = skimage.io.imread(filename)

    # Run detection
    results = model.detect([image], verbose=1)
    
    another_dict = {}

    # Visualize results
    r = results[0]
    result_dict = visualize2.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ishaan/foodtrack_cred.json"

    im = Image.open(filename)
    bucket = storage.bucket()
    for i in result_dict:
        x1 = i['x1']
        y1 = i['y1']
        x2 = i['x2']
        y2 = i['y2']
        label = i['label']
        confidence = i['confidence']

        labelname = label + str(confidence) + '.jpg'

        print(x1,y1,x2,y2)
        im1 = im.crop((x1, y1, x2, y2))
        im1.save(os.path.join(IMAGE_DIR, labelname))

        blob = bucket.blob("items/dBaYXgjNNDe6kQOluLNMmC34ssi1/" + labelname)
        outfile = os.path.join(IMAGE_DIR, labelname)
        blob.upload_from_filename(outfile)

    another_dict['items'] = result_dict

    db = firestore.Client()
    db.collection('users').document('dBaYXgjNNDe6kQOluLNMmC34ssi1').update(another_dict)

    blob = bucket.blob("snapshots/dBaYXgjNNDe6kQOluLNMmC34ssi1/" + str(newfilename) + ".jpg")
    outfile = filename

    blob.upload_from_filename(outfile)
    print("ends")
    #break
    
