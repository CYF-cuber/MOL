# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.utils.data.dataloader as DataLoader
import random
import dlib
import math

import sys


from utils.align_face import align_face
from utils.compute_TVL1 import compute_TVL1

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./utils/shape_predictor_68_face_landmarks.dat')
ALIGN_SIZE = 144
CROP_SIZE = 128
LANDMARK_NUM = 68
RANDOM_CROP_SAMPLE_NUM = 5
CLIP_LENGTH = 8

def img_pre_dlib(detector,predictor,img_path,box_enlarge=2.5,img_size=ALIGN_SIZE):
    img = cv2.imread(img_path)#[:,80:560]

    img_dlib = dlib.load_rgb_image(img_path)
    dets = detector(img_dlib, 1)
    shape = predictor(img_dlib, dets[0])
    ldm = np.matrix([[p.x, p.y] for p in shape.parts()])
    ldm=ldm.reshape(136,1)

    aligned_img, new_land = align_face(img, ldm, box_enlarge, img_size)
    return aligned_img, new_land

def crop_img_ldm(img, ldm, crop_x=8, crop_y=8,crop_size = CROP_SIZE, ldm_num = LANDMARK_NUM):
    crop_img = img[crop_x:(crop_size+crop_x), crop_y:(crop_size+crop_y),:]
    crop_ldm = ldm
    for l in range(ldm_num):
        crop_ldm[2*l] = ldm[2*l]- crop_y
        crop_ldm[2*l+1] = ldm[2*l+1] - crop_x
    return crop_img, crop_ldm

class videoDataset(Dataset):
    def __init__(self, video_path, mode_train=True):
        self.mode_train = mode_train
        
        self.video_list = []
        self.ldm_list = []
        self.flow_list = []
        self.video_labels = []
        '''
        for video in video_path:
            self.load_class_video(video, load=load_data , mode_train =self.mode_train)'''
        for video in range(len(video_path)):
            self.load_class_video(video_path[video], class_num = video)
        self.video_list = np.asarray(self.video_list).astype('float32')
        self.flow_list = np.asarray(self.flow_list).astype('float32')
        self.ldm_list = np.asarray(self.ldm_list).astype('float32')
        self.training_samples = len(self.video_list)
        print(self.video_list.shape)
        print(self.flow_list.shape)
        print(self.ldm_list.shape)

        self.video_list -= np.mean(self.video_list)
        self.video_list /= np.max(self.video_list)

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, item):
        return self.video_list[item], self.flow_list[item], self.ldm_list[item], self.video_labels[item]

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]

    def video_length_judge(self, framelist):
        video_len = len(framelist)
        if video_len <CLIP_LENGTH :
            return False
        else:
            return True

    def load_video(self, video_path, framelist):

        video_len = len(framelist)

        sample_interval = video_len // CLIP_LENGTH
        sample_frames = []
        sample_flows = []
        sample_landmarks = []


        if self.mode_train:
            for j in range(sample_interval):
                frames = []
                landmarks = []
                flows = []

                for i in range(CLIP_LENGTH):
                    img_path = video_path + '/' + framelist[i * sample_interval + j]
                    align_img, align_ldm = img_pre_dlib(detector,predictor,img_path)
                    align_img = cv2.cvtColor(align_img, cv2.COLOR_BGR2GRAY)[..., None]
                    #print(align_img.shape)
                    frames.append(align_img)
                    landmarks.append(align_ldm)
                
                for _ in range(RANDOM_CROP_SAMPLE_NUM):
                    crop_frames = []
                    crop_ldms = []
                    crop_flows = []
                    delta_x = random.randint(0,ALIGN_SIZE-CROP_SIZE)
                    delta_y = random.randint(0,ALIGN_SIZE-CROP_SIZE)

                    for n in range(len(frames)):
                        crop_img, crop_ldm  = crop_img_ldm(frames[n], landmarks[n], crop_x= delta_x, crop_y= delta_y)
                        crop_frames.append(crop_img)
                        crop_ldms.append(crop_ldm)

                    for f in range(len(crop_frames) - 1):
                        flow = compute_TVL1(crop_frames[f], crop_frames[f + 1])
                        crop_flows.append(flow)

                    sample_frames.append(crop_frames)
                    sample_landmarks.append(crop_ldms)
                    sample_flows.append(crop_flows)

        
            sample_frames = np.asarray(sample_frames)
            sample_flows = np.asarray(sample_flows)
            sample_landmarks = np.asarray(sample_landmarks)
            print(sample_frames.shape)
            return sample_frames, sample_flows, sample_landmarks
            
        else:
            for j in range(sample_interval):
                frames = []
                landmarks = []
                flows = []
                for i in range(CLIP_LENGTH):
                    img_path = video_path + '/' + framelist[i * sample_interval + j]
                    align_img, align_ldm = img_pre_dlib(detector,predictor,img_path)
                    align_img = cv2.cvtColor(align_img, cv2.COLOR_BGR2GRAY)[..., None]
                    #print(align_img.shape)
                    crop_img, crop_ldm  = crop_img_ldm(align_img, align_ldm)
                    frames.append(crop_img)
                    landmarks.append(crop_ldm)

                for f in range(len(frames) - 1):
                    flow = compute_TVL1(frames[f], frames[f + 1])
                    flows.append(flow)

                sample_frames.append(frames)
                sample_landmarks.append(landmarks)
                sample_flows.append(flows)

            sample_frames = np.asarray(sample_frames)
            sample_flows = np.asarray(sample_flows)
            sample_landmarks = np.asarray(sample_landmarks)
            #print(frames.shape)
            
            return sample_frames, sample_flows, sample_landmarks

    def load_class_video(self, videos,class_num = None):

        for video in videos:

            videopath = VIDEO_LIST[class_num] + video
  
            print(videopath)
            framelist = os.listdir(videopath)
            
            if "EP" in video:
                framelist.sort(key=lambda x: int(x.split('img')[1].split('.jpg')[0])) #CASME2
            elif 's' in video:
                framelist.sort(key=lambda x: int(x.split('image')[1].split('.jpg')[0])) #SMIC
            else:
                framelist.sort(key=lambda x: int(x.split('.')[0])) #SAMM

            if self.video_length_judge( framelist) is False:
                continue

            video_arrays, flow_arrays, landmark_arrays = self.load_video(videopath, framelist)
            sample_num, T, H, W, C = video_arrays.shape
            for sam in range(sample_num):
                video_array = video_arrays[sam]
                flow_array = flow_arrays[sam]
                landmark_array = landmark_arrays[sam]
                video_array = np.rollaxis(video_array,3,0)
                flow_array = np.rollaxis(flow_array,3,0)
                landmark_array = np.rollaxis(landmark_array,1,0)
                #print(video_array.shape)

                if len(video_array) <= 0:
                    print("video invalid!")
                    continue
                self.video_list.append(video_array)
                self.flow_list.append(flow_array)
                self.ldm_list.append(landmark_array)
                self.video_labels.append(class_num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="SAMM")
    parser.add_argument('--cls', type=int, default=3)
    parser.add_argument('--mode_train',type=bool, default=True)
    parser.add_argument('--net_test',action='store_true')# bug test
    args = parser.parse_args()
    
    if args.dataset == "CASME2" and args.cls== 3:
        surprisepath_c = './data/CASME2_data_3/surprise/'
        positivepath_c = './data/CASME2_data_3/positive/'
        negativepath_c = './data/CASME2_data_3/negative/'
        VIDEO_LIST = [surprisepath_c, positivepath_c, negativepath_c]
        LOSO = ['17', '26', '16', '09', '05', '24', '02', '13', '04', '23', '11', '12', '08', '14', '03', '19', '01', '10', '20', '21', '22', '15', '06', '25', '07']

    if args.dataset == "SAMM" and args.cls== 3:
        surprisepath_s = './data/SAMM_data_3/surprise/'
        positivepath_s = './data/SAMM_data_3/positive/'
        negativepath_s = './data/SAMM_data_3/negative/'
        VIDEO_LIST = [surprisepath_s, positivepath_s, negativepath_s]
        LOSO =['006','007','009','010','011','012','013','014','015','016','017','018','019','020','021','022','023','024','026','028','030','031','032','033','034','035','036','037']

    if args.dataset == "SMIC":
        surprisepath_sm = './data/SMIC_data_3/surprise/'
        positivepath_sm = './data/SMIC_data_3/positive/'
        negativepath_sm = './data/SMIC_data_3/negative/'
        VIDEO_LIST = [surprisepath_sm, positivepath_sm, negativepath_sm]
        LOSO = ['s1','s2','s3','s4','s5','s6','s8','s9','s11','s12','s13','s14','s15','s18','s19','s20']
      
    if args.dataset == "CASME2" and args.cls== 5:
        surprise_path = './data/CASME2_data_5/surprise/'
        happiness_path = './data/CASME2_data_5/happiness/'
        disgust_path = './data/CASME2_data_5/disgust/'
        repression_path = './data/CASME2_data_5/repression/'
        others_path = './data/CASME2_data_5/others/'
        VIDEO_LIST = [surprise_path , happiness_path, disgust_path , repression_path , others_path]
        LOSO = ['01', '17', '26', '16', '09', '05', '24', '02', '13', '04', '23', '11', '12', '08', '14', '03', '19', '10', '20', '21', '22', '15', '06', '25', '07']

    if args.dataset == "SAMM" and args.cls== 5:
        surprise_path = './data/SAMM_data_5/surprise/'
        happiness_path = './data/SAMM_data_5/happiness/'
        anger_path = './data/SAMM_data_5/anger/'
        contempt_path = './data/SAMM_data_5/contempt/'
        others_path = './data/SAMM_data_5/others/'
        VIDEO_LIST = [surprise_path , happiness_path, anger_path , contempt_path , others_path]

        LOSO =['006','007','009','010','011','012','013','014','015','016','017','018','019','020','021','022','023','025','026','028','030','031','032','033','034','035','036','037']


    videos = [os.listdir(i) for i in VIDEO_LIST]
    
    for sub in range(len(LOSO)):

        subject = LOSO[sub]
        if args.cls == 3:
            dataset_list = [[],[],[]]
        elif args.cls == 5:
            dataset_list = [[],[],[],[],[]]
        else :
            print('aug cls is invalid!')
            break

        for cla in range(len(videos)):
            class_video = videos[cla]
            for v in class_video:
                if v.split('_')[0] == subject:
                    dataset_list[cla].append(v)

        #print(dataset_list)
        if args.mode_train:
            mode = 'train'
        else:
            mode = 'test'
        save_path = 'data_processed/'+args.dataset+'/sub'+subject+'_'+str(args.cls)+'cls_'+mode+'.pth'
        print(save_path)

        #print(train_list)
        if args.net_test:
            dataset_list = [c[0:1] for c in dataset_list]

        #print(dataset_list)
        dataset =  videoDataset(dataset_list)

        torch.save(dataset, save_path)
