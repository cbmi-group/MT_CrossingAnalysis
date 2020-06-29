import cv2
import numpy as np
import math


def img_PreProc(img, img_type):
    if img_type == 8:
        img = img_clahe(img)
        img = img / 255.
        img = img_normalized(img)
    elif img_type == 16:
        img = img_clahe(img)
        img = img / 65535.
        img = img_normalized(img)
    o_img = img
    return o_img

def img_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    return img

def img_normalized(img):
    std = np.std(img)
    mean = np.mean(img)
    img_normalized = (img - mean) / (std + 1e-10)
    return img_normalized

def padding(img):
    h,w = img.shape
    assert h <= 1024 and w <= 1024, 'Image size better not larger than 1024x1024'

    log_h = int(math.log(h, 2))
    log_w = int(math.log(w, 2))
    if log_h != math.log(h, 2) and log_w != math.log(w, 2):
        padding_h = int(np.power(2, log_h + 1) - h)
        padding_w = int(np.power(2, log_w + 1) - w)
        p_h = np.zeros((padding_h, w))
        p_w = np.zeros((h + padding_h, padding_w))
        p_img = np.concatenate((img, p_h), axis=0)
        p_img = np.concatenate((p_img, p_w), axis=1)
        output = p_img
    elif log_h != math.log(h, 2) and log_w == math.log(w, 2):
        padding_h = int(np.power(2, log_h + 1) - h)
        p_h = np.zeros((padding_h, w))
        p_img = np.concatenate((img, p_h), axis=0)
        output = p_img
    elif log_h == math.log(h, 2) and log_w != math.log(w, 2):
        padding_w = int(np.power(2, log_w + 1) - w)
        p_w = np.zeros((h, padding_w))
        p_img = np.concatenate((img, p_w), axis=1)
        output = p_img
    else:
        output = img
    return output

