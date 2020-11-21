#
# Copyright 2019 Subhojeet Pramanik, Aman Husain, Priyanka Agrawal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ======================================================================
"""
Authors: Subhojeet Pramanik

OmniNet prediction script.

"""
import os
import argparse
import pickle
import json
import cv2
import torch
import numpy as np
import libs.omninet as omninet
import sys, os
import time
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
from torchvision import transforms

# Get pytorch tensor size in mb
def get_tensor_size(t):
    return t.element_size() * t.nelement() / (1024 ** 2.)

def extract_frames_from_video(video_file, EXTRACT_FREQUENCY=4, video_resize_height=300,
                              video_resize_width=300, crop_size=224, clip_len=16):
    capture=cv2.VideoCapture(video_file)
    fps = capture.get(cv2.CAP_PROP_FPS)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video fps is {fps}, frame width is {frame_width}, frame height is {frame_height}")

    if frame_count // EXTRACT_FREQUENCY <= 16:
        EXTRACT_FREQUENCY -= 1
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
    count = 0
    i = 0
    retaining = True
    frames=[]
    while (count < frame_count and retaining):
        retaining, frame = capture.read()
        if frame is None:
            continue

        if count % EXTRACT_FREQUENCY == 0:
            if (frame_height != video_resize_height) or (frame_width != video_resize_width):
                frame = cv2.resize(frame, (video_resize_width, video_resize_height))
            frames.append(frame)
            i += 1
        count += 1
    capture.release()

    total_frame_count = frame_count
    frame_count = len(frames)
    print(f'Total frame count is {total_frame_count} and {frame_count} is selected')

    buffer = np.empty((frame_count, video_resize_height, video_resize_width, 3), np.dtype('float32'))
    for i, frame in enumerate(frames):
        buffer[i] = np.array(frame)
    time_index=0
    height_index=0
    width_index=0

    buffer = buffer[time_index:time_index + clip_len,
                height_index:height_index + crop_size,
                width_index:width_index + crop_size, :]
    #Normalize
    buffer=buffer/255
    for i, frame in enumerate(buffer):
        frame -= np.array([[[0.485, 0.456, 0.406]]])
        frame /= np.array([[[0.229, 0.224, 0.225]]])
        buffer[i] = frame
    buffer=buffer.transpose((0, 3, 1, 2))
    buffer=torch.from_numpy(buffer)
    return buffer.unsqueeze(0)

def extract_pixels_from_image(image):
    img = Image.open(image)
    img = img.convert('RGB')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tfs=transforms.Compose([
                                    transforms.Resize(int(224*1.14)),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize,
                                ])
    img=tfs(img)
    img=img.unsqueeze(0)
    return img


def vision_and_language_prediction(cfg, task, image=None, text=None, video=None):

    model_file=cfg.OMNINET.MODEL
    verbose=cfg.OMNINET.VERBOSE
    penn_vocab_file = os.path.join(cfg.OMNINET.BASE, 'conf/penn_vocab.json')
    vqa_vocab_file = os.path.join(cfg.OMNINET.BASE, 'conf/vqa_vocab.pkl')
    hmdb_labels_file = os.path.join(cfg.OMNINET.BASE, 'conf/hmdblabels.txt')

    if verbose==False:
        sys.stdout = open(os.devnull, 'w')

    #Load Omninet model
    model = omninet.OmniNet(gpu_id=0)
    model.restore_file(model_file)
    model=model.to(0)
    model=model.eval()
    model.reset(1)

    if image is not None:
        image=extract_pixels_from_image(image)
        print(f'Image encoding input tensor shape is {image.size()}')
        print(f'Image encoding input tensor size is {get_tensor_size(image_encodings):.3f}')
        image=image.to(0)

        image_start = time.time()
        model.encode_images(image)
        print(f'Encode image took {time.time() - image_start}')


    if text is not None:
        text_start = time.time()
        model.encode_englishtexts([text])
        print(f'Encode text took {time.time() - text_start}')

    if video is not None:
        video=extract_frames_from_video(video, cfg.OMNINET.EXTRACT_FREQUENCY, cfg.OMNINET.VIDEO_RESIZE_HEIGHT,
                                        cfg.OMNINET.VIDEO_RESIZE_WIDTH, cfg.OMNINET.CROP_SIZE, cfg.OMNINET.CLIP_LEN)
        print(f'Video encoding input tensor shape is {video.size()}')
        print(f'Video encoding input tensor size is {get_tensor_size(video):.3f}')
        video=video.to(0)

        video_start = time.time()
        model.encode_videos(video)
        print(f'Encode videos took {time.time() - video_start}')

    if verbose == False:
        sys.setdout = sys.__stdout__


    result = ""
    start = time.time()
    if task=='caption':
        prediction=model.decode_greedy('IMAGE_CAPTION',num_steps=100)
        prediction = prediction.argmax(-1)
        prediction = model.english_language_perph.decode_tokens(prediction)
        result += f'Caption Prediction: {prediction[0]}'

    elif task=='hmdb':
        prediction = model.decode_greedy('HMDB', num_steps=1)
        prediction = prediction.argmax(-1).cpu().tolist()[0][0]
        with open(hmdb_labels_file,'r') as  f:
            lines=f.readlines()
        id_to_label=dict()
        for l in lines:
            id,label=l.split(' ')
            id_to_label[id]=label
        prediction=id_to_label[str(prediction)]
        result += f'Action recognition prediction: {prediction}'

    elif task=='vqa':
        prediction = model.decode_greedy('VQA', num_steps=1)
        prediction = prediction.argmax(-1).cpu().tolist()[0][0]
        with open(vqa_vocab_file,'rb') as f:
            ans_to_id,id_to_ans=pickle.loads(f.read())
        prediction=id_to_ans[prediction]
        result += f'VQA Prediction: {prediction}'

    elif task=='penn':
        if text is None:
            raise Exception('No text has been provided. POS tagging cannot proceed.')
        prediction= model.decode_greedy('PENN', num_steps=len(text.split(' ')))
        prediction=prediction.argmax(-1).cpu().tolist()[0]
        with open(penn_vocab_file,'r') as f:
            data=json.loads(f.read())
        id_to_tag=data['id_to_tag']
        penn_text=''
        for p in prediction:
            penn_text='%s %s'%(penn_text,id_to_tag[str(p)])
            result += f'POS tagging Prediction: {penn_text}'
    print(f'inference took {time.time() - start}')
    return result


        

        

        