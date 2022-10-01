# Import Libraries
#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
import constants as c
import os, sys
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from datetime import datetime, date, time, timezone
import threading
import ntpath
import cv2
import tkinter as tk
import imageio.v2 as imageio
import os
import time
import sys
from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia, QtMultimediaWidgets
import numpy as np
from PyQt5.QtWidgets import *
import sys
import PyQt5
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget, QGraphicsVideoItem
import ctypes
import time
import os
import subprocess
import platform
import glob
import csv
from tkinter import *
from tkinter.ttk import *
import sys
import base64
import pyautogui
import pandas as pd
import random
import math
import time
from PyQt5.QtWidgets import *
import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import ctypes
import tobii_research as tr
import time
import os
import subprocess
import platform
import glob
import csv
from tkinter import *
import sys
import base64
from datetime import datetime
import scipy.signal as signal
import statistics as stat
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from math import ceil
import constants as c
from helpers import *

global_gaze_data = None  # Initializes Gaze Data

def draw2():
    """
    Draws visualization on video right after experiment
    """
    global filepath
    global filtered_eye_data
    df = pd.read_csv(filtered_eye_data, encoding='ISO-8859-7')
    cap = cv2.VideoCapture(filepath)
    buttonBack5.place_forget()
    x = df['Left_Gaze_Point_On_Display_Area_X'].tolist()
    y = df['Left_Gaze_Point_On_Display_Area_Y'].tolist()
    x2 = df['Right_Gaze_Point_On_Display_Area_X'].to_list()
    y2 = df['Right_Gaze_Point_On_Display_Area_Y'].to_list()
    eye_time1 = df['Devise_Time'].tolist()
    eye_time = []
    sub = eye_time1[0]
    for i in range(len(eye_time1)):
        eye_time1[i] = int((eye_time1[i] - sub) / 1000)
    for i in range(len(eye_time1)):
        if eye_time1[i] >= 0:
            eye_time.append(eye_time1[i])
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_ratio = frame_width/frame_height
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    y_dim = c.Y_DIM
    x_dim = y_dim * frame_ratio
    factor_x = x_dim / frame_width
    factor_y = y_dim / frame_height
    y_center = c.Y_CENTER
    x_center = 960
    r1 = 8
    r2 = 6
    if frame_height == c.RESOLUTION:
        r1 = 9
        r2 = 7
    vid_name = str(project_directory) + '/VideoGaze.avi'
    output = cv2.VideoWriter(
        vid_name, cv2.VideoWriter_fourcc(*'DIVX'),
        fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if ret:
            # Adding filled rectangle on each frame
            video_time = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            x_draw = None
            y_draw = None
            x2_draw = None
            y2_draw = None
            for i in range(len(eye_time)):
                if eye_time[i] < video_time and i + 64 < len(eye_time):
                    x_draw = x[i + 64]
                    y_draw = y[i + 64]
                    x2_draw = x2[i + 64]
                    y2_draw = y2[i + 64]
            if x_draw != None and y_draw != None and x_draw >= x_center - x_dim / 2 and x_draw <= x_center + x_dim / 2 and y_draw >= y_center - y_dim / 2 and y_draw <= y_center + y_dim / 2:
                x_draw = x_draw - (x_center - x_dim / 2)
                y_draw = y_draw - (y_center - y_dim / 2)
                x2_draw = x2_draw - (x_center - x_dim / 2)
                y2_draw = y2_draw - (y_center - y_dim / 2)
                cv2.circle(frame, (int(x_draw / factor_x), int(y_draw / factor_y)), r1, c.GREEN_VID, -1)
                cv2.circle(frame, (int(x_draw / factor_x), int(y_draw / factor_y)), r2, c.PURPLE_VID, -1)
                cv2.circle(frame, (int(x2_draw / factor_x), int(y2_draw / factor_y)), r1, c.GREEN_VID, -1)
                cv2.circle(frame, (int(x2_draw / factor_x), int(y2_draw / factor_y)), r2, c.PURPLE_VID, -1)
                output.write(frame)
                cv2.imshow("output", frame)
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break
                if cv2.getWindowProperty("output", cv2.WND_PROP_VISIBLE) < 1:
                    sys.exit(0)
            else:
                output.write(frame)
                cv2.imshow("output", frame)
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break
                if cv2.getWindowProperty("output", cv2.WND_PROP_VISIBLE) < 1:
                    sys.exit(0)
        else:
            break

    cv2.destroyAllWindows()
    output.release()
    cap.release()
    end()


def draw_vid():
    """
    Draws visualization on video with saved data
    """
    # Reading the input
    label.config(text=" ")
    buttonVideo.place_forget()
    buttonImage.place_forget()

    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    df = pd.read_csv(root.filename_csv, encoding='ISO-8859-7')
    cap = cv2.VideoCapture(root.filename_vid)
    buttonBack5.place_forget()
    files = df['Filepath']

    if isinstance(files[0], float):
        if date_finish == None and date_start == None:
            x = df['Left_Gaze_Point_On_Display_Area_X'].tolist()
            y = df['Left_Gaze_Point_On_Display_Area_Y'].tolist()
            x2 = df['Right_Gaze_Point_On_Display_Area_X'].to_list()
            y2 = df['Right_Gaze_Point_On_Display_Area_Y'].to_list()
            eye_time1 = df['Devise_Time'].tolist()
        else:
            x = []
            y = []
            x2 = []
            y2 = []
            eye_time1 = []
            x_l = df['Left_Gaze_Point_On_Display_Area_X'].tolist()
            y_l = df['Left_Gaze_Point_On_Display_Area_Y'].tolist()
            x_r = df['Right_Gaze_Point_On_Display_Area_X'].to_list()
            y_r = df['Right_Gaze_Point_On_Display_Area_Y'].to_list()
            eye_t = df['Devise_Time'].tolist()
            date_time = df['Date_time'].tolist()
            for i in range(len(date_time)):
                if date_time[i] > date_time[index_start] and date_time[i] < date_time[index_stop]:
                    x.append(x_l[i])
                    y.append(y_l[i])
                    x2.append(x_r[i])
                    y2.append(y_r[i])
                    eye_time1.append(eye_t[i])
            if len(x) == 0:
                label.config(text="Εισάγεται Εικόνα/Βίντεο")
                sys.exit(0)
        eye_time = []
        sub = eye_time1[0]
        for i in range(len(eye_time1)):
            eye_time1[i] = int((eye_time1[i] - sub) / 1000)
        for i in range(len(eye_time1)):
            if eye_time1[i] >= 0:
                eye_time.append(eye_time1[i])
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        r1 = 8
        r2 = 6
        if frame_height == c.RESOLUTION:
            r1 = 9
            r2 = 7

        width = user32.GetSystemMetrics(0)
        height = user32.GetSystemMetrics(1)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        out_name = uniquify(str(home_directory) + "/" + "VideoGaze.avi")
        output = cv2.VideoWriter(
            out_name, cv2.VideoWriter_fourcc(*'DIVX'),
            fps, (frame_width, frame_height))
        while True:
            ret, frame = cap.read()
            if ret:
                # Adding filled rectangle on each frame
                video_time = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                x_draw = None
                y_draw = None
                x2_draw = None
                y2_draw = None
                for i in range(len(eye_time)):
                    if eye_time[i] < video_time and i < len(eye_time):
                        x_draw = x[i]
                        y_draw = y[i]
                        x2_draw = x2[i]
                        y2_draw = y2[i]
                if x_draw != None and y_draw != None and x_draw > 0 and x_draw <= width and y_draw >= 0 and y_draw <= height and \
                        x2_draw != None and y2_draw != None and x2_draw > 0 and x2_draw <= width and y2_draw >= 0 and y2_draw <= height and video_time < \
                        eye_time[-1]:
                    cv2.circle(frame, (int(x_draw), int(y_draw)), r1, c.GREEN_VID, -1)
                    cv2.circle(frame, (int(x_draw), int(y_draw)), r2, c.PURPLE_VID, -1)
                    cv2.circle(frame, (int(x2_draw), int(y2_draw)), r1, c.GREEN_VID, -1)
                    cv2.circle(frame, (int(x2_draw), int(y2_draw)), r2, c.PURPLE_VID, -1)
                    output.write(frame)
                    cv2.imshow("output", frame)
                    if cv2.waitKey(1) & 0xFF == ord('s'):
                        break
                    if cv2.getWindowProperty("output", cv2.WND_PROP_VISIBLE) < 1:
                        sys.exit(0)
                elif x_draw == None or y_draw == None or x_draw >= width or x_draw < 0 or y_draw >= height or y_draw < 0 or \
                        x2_draw == None or y2_draw == None or x2_draw >= width or x2_draw < 0 or y2_draw >= height or y2_draw < 0 or                    \
                        eye_time[-1] < video_time:
                    output.write(frame)
                    cv2.imshow("output", frame)
                    if cv2.waitKey(1) & 0xFF == ord('s'):
                        break
                    if cv2.getWindowProperty("output", cv2.WND_PROP_VISIBLE) < 1:
                        sys.exit(0)
            else:
                break

        cv2.destroyAllWindows()
        output.release()
        cap.release()
    elif isinstance(files[0], str):
        width = user32.GetSystemMetrics(0)
        height = user32.GetSystemMetrics(1)
        x = df['Left_Gaze_Point_On_Display_Area_X'].tolist()
        y = df['Left_Gaze_Point_On_Display_Area_Y'].tolist()
        x2 = df['Right_Gaze_Point_On_Display_Area_X'].tolist()
        y2 = df['Right_Gaze_Point_On_Display_Area_Y'].tolist()
        eye_time1 = df['Devise_Time'].tolist()
        eye_time = []
        sub = eye_time1[0]
        for i in range(len(eye_time1)):
            eye_time1[i] = int((eye_time1[i] - sub) / 1000)
        for i in range(len(eye_time1)):
            if eye_time1[i] >= 0:
                eye_time.append(eye_time1[i])
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_ratio = frame_width / frame_height
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        y_dim = c.Y_DIM
        x_dim = y_dim * frame_ratio
        factor_x = x_dim / frame_width
        factor_y = y_dim / frame_height
        y_center = c.Y_CENTER
        x_center = 960
        r1 = 8
        r2 = 6
        if frame_height == c.RESOLUTION:
            r1 = 9
            r2 = 7
        out_name = uniquify(str(home_directory) + "/" + "VideoGaze.avi")
        output = cv2.VideoWriter(
            out_name, cv2.VideoWriter_fourcc(*'DIVX'),
            fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if ret:
                # Adding filled rectangle on each frame
                video_time = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                x_draw = None
                y_draw = None
                x2_draw = None
                y2_draw = None
                for i in range(len(eye_time)):
                    if eye_time[i] < video_time and i + 64 < len(eye_time):
                        x_draw = x[i + 64]
                        y_draw = y[i + 64]
                        x2_draw = x2[i + 64]
                        y2_draw = y2[i + 64]
                if x_draw != None and y_draw != None and x_draw > 0 and x_draw <= width and y_draw >= 0 and y_draw <= height and \
                        x2_draw != None and y2_draw != None and x2_draw > 0 and x2_draw <= width and y2_draw >= 0 and y2_draw <= height and video_time < \
                        eye_time[-1]:
                    x_draw = x_draw - (x_center - x_dim / 2)
                    y_draw = y_draw - (y_center - y_dim / 2)
                    x2_draw = x2_draw - (x_center - x_dim / 2)
                    y2_draw = y2_draw - (y_center - y_dim / 2)

                    cv2.circle(frame, (int(x_draw / factor_x), int(y_draw / factor_y)), r1, c.GREEN_VID, -1)
                    cv2.circle(frame, (int(x_draw / factor_x), int(y_draw / factor_y)), r2, c.PURPLE_VID, -1)
                    cv2.circle(frame, (int(x2_draw / factor_x), int(y2_draw / factor_y)), r1, c.GREEN_VID, -1)
                    cv2.circle(frame, (int(x2_draw / factor_x), int(y2_draw / factor_y)), r2, c.PURPLE_VID, -1)
                    output.write(frame)
                    cv2.imshow("output", frame)
                    if cv2.waitKey(1) & 0xFF == ord('s'):
                        break
                    if cv2.getWindowProperty("output", cv2.WND_PROP_VISIBLE) < 1:
                        sys.exit(0)
                elif x_draw == None or y_draw == None or x_draw >= width or x_draw < 0 or y_draw >= height or y_draw < 0 or \
                            x2_draw == None or y2_draw == None or x2_draw >= width or x2_draw < 0 or y2_draw >= height or y2_draw < 0 or \
                            eye_time[-1] < video_time:
                    output.write(frame)
                    cv2.imshow("output", frame)
                    if cv2.waitKey(1) & 0xFF == ord('s'):
                        break
                    if cv2.getWindowProperty("output", cv2.WND_PROP_VISIBLE) < 1:
                        sys.exit(0)
            else:
                break

        cv2.destroyAllWindows()
        output.release()
        cap.release()

    button2.place_forget()
    button1.place_forget()
    label.place(relheight=1, relwidth=1)
    label.config(text='Αρχικό μενού')
    buttonEye.place(relx=0.4, rely=0, relwidth=0.2, relheight=0.4)
    buttonData.place(relx=0.4, rely=0.5, relwidth=0.2, relheight=0.4)



def FixationMapIm(x_list, y_list, time_list, imagen):
    image = imageio.imread(imagen)
    overlay = image.copy()
    alpha = c.ALPHA_FIX
    radius = c.RADIUS_FIX
    if len(x_list) == 0:
        name = uniquify(str(home_directory) + '/' + 'Fixation_Map.png')
        cv2.imwrite(name, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        min_time = min(time_list)
        max_time = max(time_list)
        differece_times = ceil(max_time - min_time)
        for i in range(len(x_list)):
            if differece_times != 0:
                radius = 25 + int(50 * (time_list[i] - min_time) / differece_times)
            else:
                radius = 25
            cv2.circle(overlay, (x_list[i], y_list[i]), radius, c.GREEN_F, -1)
            cv2.circle(overlay, (x_list[i], y_list[i]), radius, c.BLACK_F, 2)
            image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        name = uniquify(str(home_directory) + '/' + 'Fixation_Map.png')
        cv2.imwrite(name, cv2.cvtColor(image_new, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def FixationScanIm(x_list, y_list, time_list, imagen):
    """
    Draws Fixation-Scanpath on image with saved data
    :param x_list: list with x coordinates of fixations
    :param y_list: list with y coordinates of fixations
    :param time_list: list with time of fixations
    :param imagen: image shown
    """
    image = imageio.imread(imagen)
    overlay = image.copy()
    # For circle
    alpha = c.ALPHA_FIX
    radius = []
    # For Arrow
    color = c.BLUE_FS
    thickness = c.THICK_FS
    if len(x_list) == 0:
        name = uniquify(str(home_directory) + '/' + 'Fixation_Map.png')
        cv2.imwrite(name, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        min_time = min(time_list)
        max_time = max(time_list)
        differece_times = ceil(max_time - min_time)
        for i in range(len(x_list)):
            if differece_times != 0:
                radius.append(25 + int(50 * (time_list[i] - min_time) / differece_times))
            else:
                radius.append(25)
        for i in range(len(x_list)):
            cv2.circle(overlay, (x_list[i], y_list[i]), radius[i], c.GREEN_F, -1)
            cv2.circle(overlay, (x_list[i], y_list[i]), radius[i], c.BLACK_F, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            if i < 10:
                cv2.putText(overlay, str(i + 1), (x_list[i] - 5, y_list[i] + 5), font, 0.5, c.BLACK_F, 2, cv2.LINE_AA)
            elif i > 9 and i < 100:
                cv2.putText(overlay, str(i + 1), (x_list[i] - 10, y_list[i] + 5), font, 0.5, c.BLACK_F, 2, cv2.LINE_AA)
            elif i > 100:
                cv2.putText(overlay, str(i + 1), (x_list[i] - 15, y_list[i] + 5), font, 0.5, c.BLACK_F, 2, cv2.LINE_AA)
            image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            if i < len(x_list) - 1:
                if x_list[i + 1] == x_list[i]:
                    theta = 90
                    if y_list[i + 1] > y_list[i]:
                        image_new = cv2.arrowedLine(overlay, (int(x_list[i]), int(y_list[i] + radius[i])),
                                                    (int(x_list[i]), int(y_list[i + 1] - radius[i + 1])),
                                                    color, thickness, tipLength=0.05)
                    elif y_list[i + 1] < y_list[i]:
                        image_new = cv2.arrowedLine(overlay, (int(x_list[i]), int(y_list[i] - radius[i])),
                                                    (int(x_list[i + 1]), int(y_list[i + 1] + radius[i + 1])),
                                                    color, thickness, tipLength=0.05)
                elif y_list[i + 1] == y_list[i]:
                    theta = 0
                    if x_list[i] < x_list[i + 1]:
                        image_new = cv2.arrowedLine(overlay, (int(x_list[i] + radius[i]), int(y_list[i])),
                                                    (int(x_list[i + 1] - radius[i + 1]), int(y_list[i + 1])),
                                                    color, thickness, tipLength=0.05)
                    elif x_list[i + 1] < x_list[i]:
                        image_new = cv2.arrowedLine(overlay, (int(x_list[i] + radius[i]), int(y_list[i])),
                                                    (int(x_list[i + 1] - radius[i + 1]), int(y_list[i + 1])),
                                                    color, thickness, tipLength=0.05)
                else:
                    theta = math.atan(abs((y_list[i + 1] - y_list[i]) / (x_list[i + 1] - x_list[i])))
                    if y_list[i + 1] < y_list[i] and x_list[i + 1] > x_list[i]:
                        image_new = cv2.arrowedLine(overlay,
                                                    (int((x_list[i] + radius[i] * math.cos(theta))),
                                                     int(y_list[i] - radius[i] * math.sin(theta))),
                                                    (
                                                        int(x_list[i + 1] - radius[i + 1] * math.cos(theta)),
                                                        int(y_list[i + 1] + radius[i + 1] * math.sin(theta))),
                                                    color, thickness, tipLength=0.05)
                    elif y_list[i + 1] > y_list[i] and x_list[i + 1] > x_list[i]:
                        image_new = cv2.arrowedLine(overlay,
                                                    (int((x_list[i] + radius[i] * math.cos(theta))),
                                                     int(y_list[i] + radius[i] * math.sin(theta))),
                                                    (
                                                        int(x_list[i + 1] - radius[i + 1] * math.cos(theta)),
                                                        int(y_list[i + 1] - radius[i + 1] * math.sin(theta))),
                                                    color, thickness, tipLength=0.05)
                    elif y_list[i + 1] > y_list[i] and x_list[i + 1] < x_list[i]:
                        image_new = cv2.arrowedLine(overlay,
                                                    (int((x_list[i] - radius[i] * math.cos(theta))),
                                                     int(y_list[i] + radius[i] * math.sin(theta))),
                                                    (
                                                        int(x_list[i + 1] + radius[i + 1] * math.cos(theta)),
                                                        int(y_list[i + 1] - radius[i + 1] * math.sin(theta))),
                                                    color, thickness, tipLength=0.05)
                    elif y_list[i + 1] < y_list[i] and x_list[i + 1] < x_list[i]:
                        image_new = cv2.arrowedLine(overlay,
                                                    (int((x_list[i] - radius[i] * math.cos(theta))),
                                                     int(y_list[i] - radius[i] * math.sin(theta))),
                                                    (
                                                        int(x_list[i + 1] + radius[i + 1] * math.cos(theta)),
                                                        int(y_list[i + 1] + radius[i + 1] * math.sin(theta))),
                                                    color, thickness, tipLength=0.05)


        name = uniquify(str(home_directory) + '/' + 'Fixation_Map.png')
        cv2.imwrite(name, cv2.cvtColor(image_new, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def FixationScan(x_list, y_list, time_list, imagen):
    """
        Draws Fixation-Scanpath on image right after experiment
        :param x_list: list with x coordinates of fixations
        :param y_list: list with y coordinates of fixations
        :param time_list: list with time of fixations
        :param imagen: image shown
    """
    image = imageio.imread(imagen)
    overlay = image.copy()
    # For Circle
    alpha = c.ALPHA_FIX
    radius = []
    # For Arrow
    color = c.BLUE_FS
    thickness = c.THICK_FS


    if len(x_list) == 0:
        name = str(fixation_dir) + '/' + str(file_stem) + '_Fixation_Map' + '.png'
        cv2.imwrite(name, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        min_time = min(time_list)
        max_time = max(time_list)
        differece_times = ceil(max_time - min_time)

        for i in range(len(x_list)):
            if differece_times != 0:
                radius.append(25 + int(50 * (time_list[i] - min_time) / differece_times))
            else:
                radius.append(25)
        for i in range(len(x_list)):
            cv2.circle(overlay, (x_list[i], y_list[i]), radius[i], c.GREEN_F, -1)
            cv2.circle(overlay, (x_list[i], y_list[i]), radius[i], c.BLACK_F, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            if i < 10:
                cv2.putText(overlay, str(i + 1), (x_list[i] - 5, y_list[i] + 5), font, 0.5, c.BLACK_F, 2, cv2.LINE_AA)
            elif i > 9 and i < 100:
                cv2.putText(overlay, str(i + 1), (x_list[i] - 10, y_list[i] + 5), font, 0.5, c.BLACK_F, 2, cv2.LINE_AA)
            elif i > 100:
                cv2.putText(overlay, str(i + 1), (x_list[i] - 15, y_list[i] + 5), font, 0.5, c.BLACK_F, 2, cv2.LINE_AA)
            image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            if i < len(x_list) - 1:
                if x_list[i + 1] == x_list[i]:
                    theta = 90
                    if y_list[i + 1] > y_list[i]:
                        image_new = cv2.arrowedLine(overlay, (int(x_list[i]), int(y_list[i] + radius[i])),
                                                    (int(x_list[i]), int(y_list[i + 1] - radius[i + 1])),
                                                    color, thickness, tipLength=0.05)
                    elif y_list[i + 1] < y_list[i]:
                        image_new = cv2.arrowedLine(overlay, (int(x_list[i]), int(y_list[i] - radius[i])),
                                                    (int(x_list[i + 1]), int(y_list[i + 1] + radius[i + 1])),
                                                    color, thickness, tipLength=0.05)
                elif y_list[i + 1] == y_list[i]:
                    theta = 0
                    if x_list[i] < x_list[i + 1]:
                        image_new = cv2.arrowedLine(overlay, (int(x_list[i] + radius[i]), int(y_list[i])),
                                                    (int(x_list[i + 1] - radius[i + 1]), int(y_list[i + 1])),
                                                    color, thickness, tipLength=0.05)
                    elif x_list[i + 1] < x_list[i]:
                        image_new = cv2.arrowedLine(overlay, (int(x_list[i] + radius[i]), int(y_list[i])),
                                                    (int(x_list[i + 1] - radius[i + 1]), int(y_list[i + 1])),
                                                    color, thickness, tipLength=0.05)
                else:
                    theta = math.atan(abs((y_list[i + 1] - y_list[i]) / (x_list[i + 1] - x_list[i])))
                    if y_list[i + 1] < y_list[i] and x_list[i + 1] > x_list[i]:
                        image_new = cv2.arrowedLine(overlay,
                                                    (int((x_list[i] + radius[i] * math.cos(theta))),
                                                     int(y_list[i] - radius[i] * math.sin(theta))),
                                                    (
                                                        int(x_list[i + 1] - radius[i + 1] * math.cos(theta)),
                                                        int(y_list[i + 1] + radius[i + 1] * math.sin(theta))),
                                                    color, thickness, tipLength=0.05)
                    elif y_list[i + 1] > y_list[i] and x_list[i + 1] > x_list[i]:
                        image_new = cv2.arrowedLine(overlay,
                                                    (int((x_list[i] + radius[i] * math.cos(theta))),
                                                     int(y_list[i] + radius[i] * math.sin(theta))),
                                                    (
                                                        int(x_list[i + 1] - radius[i + 1] * math.cos(theta)),
                                                        int(y_list[i + 1] - radius[i + 1] * math.sin(theta))),
                                                    color, thickness, tipLength=0.05)
                    elif y_list[i + 1] > y_list[i] and x_list[i + 1] < x_list[i]:
                        image_new = cv2.arrowedLine(overlay,
                                                    (int((x_list[i] - radius[i] * math.cos(theta))),
                                                     int(y_list[i] + radius[i] * math.sin(theta))),
                                                    (
                                                        int(x_list[i + 1] + radius[i + 1] * math.cos(theta)),
                                                        int(y_list[i + 1] - radius[i + 1] * math.sin(theta))),
                                                    color, thickness, tipLength=0.05)
                    elif y_list[i + 1] < y_list[i] and x_list[i + 1] < x_list[i]:
                        image_new = cv2.arrowedLine(overlay,
                                                    (int((x_list[i] - radius[i] * math.cos(theta))),
                                                     int(y_list[i] - radius[i] * math.sin(theta))),
                                                    (
                                                        int(x_list[i + 1] + radius[i + 1] * math.cos(theta)),
                                                        int(y_list[i + 1] + radius[i + 1] * math.sin(theta))),
                                                    color, thickness, tipLength=0.05)

    file_stem = Path(imagen).stem
    name = str(fixation_dir) + '/' + str(file_stem) + '_Fixation_Map' + '.png'
    cv2.imwrite(name, cv2.cvtColor(image_new, cv2.COLOR_RGB2BGR))
    is_success, im_buf_arr = cv2.imencode(".png", cv2.cvtColor(image_new, cv2.COLOR_RGB2BGR))
    im_buf_arr.tofile(name)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def FixationMap(x_list, y_list, time_list, imagen):
    image = imageio.imread(imagen)
    overlay = image.copy()
    alpha = c.ALPHA_FIX
    radius = c.RADIUS_FIX

    if len(x_list) == 0:
        name = str(fixation_dir) + '/' + str(file_stem) + '_Fixation_Map' + '.png'
        cv2.imwrite(name, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        min_time = min(time_list)
        max_time = max(time_list)
        differece_times = ceil(max_time - min_time)

        for i in range(len(x_list)):
            if differece_times != 0:
                radius = 25 + int(50 * (time_list[i] - min_time) / differece_times)
            else:
                radius = 25
            cv2.circle(overlay, (x_list[i], y_list[i]), radius, c.GREEN_F, -1)
            cv2.circle(overlay, (x_list[i], y_list[i]), radius, c.BLACK_F, 2)
            image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        file_stem = Path(imagen).stem
        name = str(fixation_dir) + '/' + str(file_stem) + '_Fixation_Map' + '.png'
        cv2.imwrite(name, cv2.cvtColor(image_new, cv2.COLOR_RGB2BGR))
        is_success, im_buf_arr = cv2.imencode(".png", cv2.cvtColor(image_new, cv2.COLOR_RGB2BGR))
        im_buf_arr.tofile(name)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def ScanpathIm(x_list, y_list, imagen):
    image = imageio.imread(imagen)
    overlay = image.copy()

    # For circle
    alpha = c.ALPHA_S
    radius = c.RADIUS_FIX

    # For arrow
    color = c.BLUE_FS
    thickness = c.THICK_FS

    if len(x_list) == 0:
        name = uniquify(str(home_directory) + '/' + 'Scanpath.png')
        cv2.imwrite(name, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        for i in range(len(x_list)):
            if i < 10:
                cv2.circle(overlay, (x_list[i], y_list[i]), radius, c.WHITE_S, -1)
                cv2.circle(overlay, (x_list[i], y_list[i]), radius, c.BLACK_F, 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(overlay, str(i + 1), (x_list[i] - 5, y_list[i] + 5), font, 0.5, c.BLACK_F, 2, cv2.LINE_AA)
                image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            elif i > 9 and i < 100:
                cv2.circle(overlay, (x_list[i], y_list[i]), radius, c.WHITE_S, -1)
                cv2.circle(overlay, (x_list[i], y_list[i]), radius, c.BLACK_F, 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(overlay, str(i + 1), (x_list[i] - 10, y_list[i] + 5), font, 0.5, c.BLACK_F, 2, cv2.LINE_AA)
                image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            elif i > 100:
                cv2.circle(overlay, (x_list[i], y_list[i]), radius, c.WHITE_S, -1)
                cv2.circle(overlay, (x_list[i], y_list[i]), radius, c.BLACK_F, 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(overlay, str(i + 1), (x_list[i] - 15, y_list[i] + 5), font, 0.5, c.BLACK_F, 2, cv2.LINE_AA)
                image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

            if i < len(x_list) - 1:
                if x_list[i + 1] == x_list[i]:
                    theta = 90
                    if y_list[i + 1] > y_list[i]:
                        image_new = cv2.arrowedLine(overlay, (int(x_list[i]), int(y_list[i] + 25)),
                                                (int(x_list[i]), int(y_list[i + 1] - 25)),
                                                color, thickness, tipLength=0.05)
                    elif y_list[i + 1] < y_list[i]:
                        image_new = cv2.arrowedLine(overlay, (int(x_list[i]), int(y_list[i] - 25)),
                                                (int(x_list[i + 1]), int(y_list[i + 1] + 25)),
                                                color, thickness, tipLength=0.05)
                elif y_list[i + 1] == y_list[i]:
                    theta = 0
                    if x_list[i] < x_list[i + 1]:
                        image_new = cv2.arrowedLine(overlay, (int(x_list[i] + 25), int(y_list[i])),
                                                (int(x_list[i + 1] - 25), int(y_list[i + 1])),
                                                color, thickness, tipLength=0.05)
                    elif x_list[i + 1] < x_list[i]:
                        image_new = cv2.arrowedLine(overlay, (int(x_list[i] + 25), int(y_list[i])),
                                                (int(x_list[i + 1] - 25), int(y_list[i + 1])),
                                                color, thickness, tipLength=0.05)
                else:
                    theta = math.atan(abs((y_list[i + 1] - y_list[i]) / (x_list[i + 1] - x_list[i])))
                    if y_list[i + 1] < y_list[i] and x_list[i + 1] > x_list[i]:
                        image_new = cv2.arrowedLine(overlay,
                                                (int((x_list[i] + 25 * math.cos(theta))),
                                                 int(y_list[i] - 25 * math.sin(theta))),
                                                (
                                                    int(x_list[i + 1] - 25 * math.cos(theta)),
                                                    int(y_list[i + 1] + 25 * math.sin(theta))),
                                                color, thickness, tipLength=0.05)
                    elif y_list[i + 1] > y_list[i] and x_list[i + 1] > x_list[i]:
                        image_new = cv2.arrowedLine(overlay,
                                                (int((x_list[i] + 25 * math.cos(theta))),
                                                 int(y_list[i] + 25 * math.sin(theta))),
                                                (
                                                    int(x_list[i + 1] - 25 * math.cos(theta)),
                                                    int(y_list[i + 1] - 25 * math.sin(theta))),
                                                color, thickness, tipLength=0.05)
                    elif y_list[i + 1] > y_list[i] and x_list[i + 1] < x_list[i]:
                        image_new = cv2.arrowedLine(overlay,
                                                (int((x_list[i] - 25 * math.cos(theta))),
                                                 int(y_list[i] + 25 * math.sin(theta))),
                                                (
                                                    int(x_list[i + 1] + 25 * math.cos(theta)),
                                                    int(y_list[i + 1] - 25 * math.sin(theta))),
                                                color, thickness, tipLength=0.05)
                    elif y_list[i + 1] < y_list[i] and x_list[i + 1] < x_list[i]:
                        image_new = cv2.arrowedLine(overlay,
                                                (int((x_list[i] - 25 * math.cos(theta))),
                                                 int(y_list[i] - 25 * math.sin(theta))),
                                                (
                                                    int(x_list[i + 1] + 25 * math.cos(theta)),
                                                    int(y_list[i + 1] + 25 * math.sin(theta))),
                                                color, thickness, tipLength=0.05)


        name = uniquify(str(home_directory) + '/' + 'Scanpath.png')
        cv2.imwrite(name, cv2.cvtColor(image_new, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def Scanpath(x_list, y_list, imagen):
    image = imageio.imread(imagen)
    overlay = image.copy()

    # For circle
    alpha = c.ALPHA_S
    radius = c.RADIUS_FIX

    # For arrow
    color = c.BLUE_FS
    thickness = c.THICK_FS

    for i in range(len(x_list)):
        if i < 10:
            cv2.circle(overlay, (x_list[i], y_list[i]), radius, c.WHITE_S, -1)
            cv2.circle(overlay, (x_list[i], y_list[i]), radius, c.BLACK_F, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(overlay, str(i + 1), (x_list[i] - 5, y_list[i] + 5), font, 0.5, c.BLACK_F, 2, cv2.LINE_AA)
            image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        elif i > 9 and i < 100:
            cv2.circle(overlay, (x_list[i], y_list[i]), radius, c.WHITE_S, -1)
            cv2.circle(overlay, (x_list[i], y_list[i]), radius, c.BLACK_F, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(overlay, str(i + 1), (x_list[i] - 10, y_list[i] + 5), font, 0.5, c.BLACK_F, 2, cv2.LINE_AA)
            image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        elif i > 100:
            cv2.circle(overlay, (x_list[i], y_list[i]), radius, c.WHITE_S, -1)
            cv2.circle(overlay, (x_list[i], y_list[i]), radius, c.BLACK_F, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(overlay, str(i + 1), (x_list[i] - 15, y_list[i] + 5), font, 0.5, c.BLACK_F, 2, cv2.LINE_AA)
            image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        if i < len(x_list) - 1:
            if x_list[i + 1] == x_list[i]:
                theta = 90
                if y_list[i + 1] > y_list[i]:
                    image_new = cv2.arrowedLine(overlay, (int(x_list[i]), int(y_list[i] + 25)),
                                                (int(x_list[i]), int(y_list[i + 1] - 25)),
                                                color, thickness, tipLength=c.TIP_FS)
                elif y_list[i + 1] < y_list[i]:
                    image_new = cv2.arrowedLine(overlay, (int(x_list[i]), int(y_list[i] - 25)),
                                                (int(x_list[i + 1]), int(y_list[i + 1] + 25)),
                                                color, thickness, tipLength=c.TIP_FS)
            elif y_list[i + 1] == y_list[i]:
                theta = 0
                if x_list[i] < x_list[i + 1]:
                    image_new = cv2.arrowedLine(overlay, (int(x_list[i] + 25), int(y_list[i])),
                                                (int(x_list[i + 1] - 25), int(y_list[i + 1])),
                                                color, thickness, tipLength=c.TIP_FS)
                elif x_list[i + 1] < x_list[i]:
                    image_new = cv2.arrowedLine(overlay, (int(x_list[i] + 25), int(y_list[i])),
                                                (int(x_list[i + 1] - 25), int(y_list[i + 1])),
                                                color, thickness, tipLength=c.TIP_FS)
            else:
                theta = math.atan(abs((y_list[i + 1] - y_list[i]) / (x_list[i + 1] - x_list[i])))
                if y_list[i + 1] < y_list[i] and x_list[i + 1] > x_list[i]:
                    image_new = cv2.arrowedLine(overlay,
                                                (int((x_list[i] + 25 * math.cos(theta))),
                                                 int(y_list[i] - 25 * math.sin(theta))),
                                                (
                                                    int(x_list[i + 1] - 25 * math.cos(theta)),
                                                    int(y_list[i + 1] + 25 * math.sin(theta))),
                                                color, thickness, tipLength=c.TIP_FS)
                elif y_list[i + 1] > y_list[i] and x_list[i + 1] > x_list[i]:
                    image_new = cv2.arrowedLine(overlay,
                                                (int((x_list[i] + 25 * math.cos(theta))),
                                                 int(y_list[i] + 25 * math.sin(theta))),
                                                (
                                                    int(x_list[i + 1] - 25 * math.cos(theta)),
                                                    int(y_list[i + 1] - 25 * math.sin(theta))),
                                                color, thickness, tipLength=c.TIP_FS)
                elif y_list[i + 1] > y_list[i] and x_list[i + 1] < x_list[i]:
                    image_new = cv2.arrowedLine(overlay,
                                                (int((x_list[i] - 25 * math.cos(theta))),
                                                 int(y_list[i] + 25 * math.sin(theta))),
                                                (
                                                    int(x_list[i + 1] + 25 * math.cos(theta)),
                                                    int(y_list[i + 1] - 25 * math.sin(theta))),
                                                color, thickness, tipLength=c.TIP_FS)
                elif y_list[i + 1] < y_list[i] and x_list[i + 1] < x_list[i]:
                    image_new = cv2.arrowedLine(overlay,
                                                (int((x_list[i] - 25 * math.cos(theta))),
                                                 int(y_list[i] - 25 * math.sin(theta))),
                                                (
                                                    int(x_list[i + 1] + 25 * math.cos(theta)),
                                                    int(y_list[i + 1] + 25 * math.sin(theta))),
                                                color, thickness, tipLength=c.TIP_FS)

    file_stem = Path(imagen).stem
    name = str(scanpath_dir) + '/' + str(file_stem) + '_Scanpath' + '.png'
    is_success, im_buf_arr = cv2.imencode(".png", cv2.cvtColor(image_new, cv2.COLOR_RGB2BGR))
    im_buf_arr.tofile(name)
    cv2.imwrite(name, image_new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def GaussianMask(sizex, sizey, sigma=33, center=None, fix=1):
    """
    sizex  : mask width
    sizey  : mask height
    sigma  : gaussian Sd
    center : gaussian mean
    fix    : gaussian max
    return gaussian mask
    """
    x = np.arange(0, sizex, 1, float)
    y = np.arange(0, sizey, 1, float)
    x, y = np.meshgrid(x, y)

    if center is None:
        x0 = sizex // 2
        y0 = sizey // 2
    else:
        if np.isnan(center[0]) == False and np.isnan(center[1]) == False:
            x0 = center[0]
            y0 = center[1]
        else:
            return np.zeros((sizey, sizex))

    return fix * np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


def Fixpos2Densemap(fix_arr, W, H, imgfile, alpha=0.5, threshold=c.THRES_GM):
    """
    fix_arr   : fixation array number of subjects x 3(x,y,fixation)
    width     : output image width
    height    : output image height
    imgfile   : image file (optional)
    alpha     : marge rate imgfile and heatmap (optional)
    threshold : heatmap threshold(0~255)
    return heatmap
    """

    heatmap = np.zeros((H, W), np.float32)
    progress['value'] = 0
    length = fix_arr.shape[0]
    for n_subject in range(length):
        progress['value'] += 100 / length
        root.update_idletasks()
        heatmap += GaussianMask(W, H, c.SIGMOID, (fix_arr[n_subject, 0], fix_arr[n_subject, 1]),
                                )

    # Normalization
    np.seterr(invalid='ignore')
    heatmap = heatmap / np.amax(heatmap)
    heatmap = heatmap * 255
    heatmap = heatmap.astype("uint8")

    if imgfile.any():
        # Resize heatmap to imgfile shape
        h, w, _ = imgfile.shape
        heatmap = cv2.resize(heatmap, (w, h))
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Create mask
        mask = np.where(heatmap <= threshold, 1, 0)
        mask = np.reshape(mask, (h, w, 1))
        mask = np.repeat(mask, 3, axis=2)

        # Marge images
        marge = imgfile * mask + heatmap_color * (1 - mask)
        marge = marge.astype("uint8")
        marge = cv2.addWeighted(imgfile, 1 - alpha, marge, alpha, 0)
        return marge

    else:
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap


def plotting(time, x, y, plt_name):
    """
    Plots x and y by time.
    :param time: devise time
    :param x: x coordinate
    :param y: y coordinate
    :param plt_name: name of plot
    """
    plt.plot(time, x, label='X coordinate')
    plt.plot(time, y, label='Y coordinate')
    plt.xlabel('Devise Time')
    plt.ylabel('Gaze Point On Screen')
    plt.title('Gaze-Time Plot')
    plt.legend()
    plt.savefig(plt_name)
    plt.close()


# Replaces invalid left and right eye gaze points (Nones) with Linear Interpolation
def interpolation(alist):
    """
    Interpolation of Eye Data
    :param alist: list of coordinates before interpolation
    """
    if len(alist) == 1:
        print("Cannot interpolate with only one sample")
    else:
        if math.isnan(alist[0]) is True:
            j = 1
            while j < len(alist) and math.isnan(alist[j]) is True:
                j += 1
            if j < len(alist):
                alist[0] = alist[j]
            elif j == len(alist):
                print("Cannot interpolate only None Values..")
        for i in range(len(alist)):
            if math.isnan(alist[i]) is True:
                j = i + 1
                while j < len(alist) and math.isnan(alist[j]) is True:
                    j += 1
                if j == len(alist):
                    alist[i] = alist[i - 1]
                elif j < len(alist):
                    gap = alist[j] - alist[i - 1]
                    s = (devise_ts[i] - devise_ts[i - 1]) / (devise_ts[j] - devise_ts[i - 1])
                    alist[i] = alist[i - 1] + s * gap


# Fixation Extraction with I-VT algorithm. Most famous algorithm for fixation extraction out of gaze points
def ivt(eye_x, eye_y, timestamp):
    """
    Implementation of I-VT algorithm
    :param eye_x: x coordinate
    :param eye_y: y coordinate
    :param timestamp: devise time
    :return: fixation
    """
    difx = []  # X distances for eye gaze point
    dify = []  # Y distances for eye gaze point
    tdif = []  # Time Distances for eye gaze point

    # Timestamp in seconds. Coordinates in pixels
    for i in range(len(eye_x) - 1):
        difx.append(float(eye_x[i + 1]) - float(eye_x[i]))
        dify.append(float(eye_y[i + 1]) - float(eye_y[i]))
        tdif.append(float(timestamp[i + 1]) - float(timestamp[i]))

    dif = np.sqrt(np.power(difx, 2) + np.power(dify, 2))

    velocity = dif / tdif

    mvmts = []  # Stores 1 if velocity is under the velocity threshold or 0 if my velocity is above

    v_threshold = c.VEL_THRES  # Seems like a good number

    for v in velocity:
        if v < v_threshold:
            mvmts.append(1)  # v < threshold --> fixation
        else:
            mvmts.append(0)  # v > threshold --> saccade

    fixation = []
    fs = []

    # Finds A list of Lists. The Fixation
    for m in range(len(mvmts)):
        if mvmts[m] == 0:
            if len(fs) > 0:
                fixation.append(fs)
                fs = []
        else:
            fs.append(m)

    if len(fs) > 0:  # Stores Remaining Fixation at the end if there is any
        fixation.append(fs)
    return fixation


def gaze_data_callback(gdata):
    """
    Callback Function whenever there is new gaze data
    :param gdata: item that corresponds to gaze data
    """
    global global_gaze_data
    global_gaze_data = gdata

    with open(raw_data_name, 'a', newline='') as p:  # Data are saved in a new csv file
        csv_writer = csv.writer(p)

        csv_writer.writerow([gdata['device_time_stamp'], gdata['left_gaze_point_on_display_area'][0],
                             gdata['right_gaze_point_on_display_area'][0], gdata['left_gaze_point_on_display_area'][1],
                             gdata['right_gaze_point_on_display_area'][1],
                             gdata['left_gaze_point_in_user_coordinate_system'][0],
                             gdata['left_gaze_point_in_user_coordinate_system'][1],
                             gdata['left_gaze_point_in_user_coordinate_system'][2],
                             gdata['right_gaze_point_in_user_coordinate_system'][0],
                             gdata['right_gaze_point_in_user_coordinate_system'][1],
                             gdata['right_gaze_point_in_user_coordinate_system'][2],
                             gdata['left_gaze_point_validity'], gdata['right_gaze_point_validity'],
                             gdata['left_pupil_diameter'], gdata['right_pupil_diameter'],
                             gdata['left_pupil_validity'], gdata['right_pupil_validity'],
                             gdata['left_gaze_origin_in_user_coordinate_system'][0],
                             gdata['left_gaze_origin_in_user_coordinate_system'][1],
                             gdata['left_gaze_origin_in_user_coordinate_system'][2],
                             gdata['right_gaze_origin_in_user_coordinate_system'][0],
                             gdata['right_gaze_origin_in_user_coordinate_system'][1],
                             gdata['right_gaze_origin_in_user_coordinate_system'][2],
                             gdata['left_gaze_origin_in_trackbox_coordinate_system'][0],
                             gdata['left_gaze_origin_in_trackbox_coordinate_system'][1],
                             gdata['left_gaze_origin_in_trackbox_coordinate_system'][2],
                             gdata['right_gaze_origin_in_trackbox_coordinate_system'][0],
                             gdata['right_gaze_origin_in_trackbox_coordinate_system'][1],
                             gdata['right_gaze_origin_in_trackbox_coordinate_system'][2],
                             gdata['left_gaze_origin_validity'], gdata['right_gaze_origin_validity'],
                             gdata['system_time_stamp'], time.time(),
                             np.sqrt((gdata['left_gaze_point_in_user_coordinate_system'][0] -
                                      gdata['left_gaze_origin_in_user_coordinate_system'][0]) ** 2
                                     + (gdata['left_gaze_point_in_user_coordinate_system'][1] -
                                        gdata['left_gaze_origin_in_user_coordinate_system'][1]) ** 2
                                     + (gdata['left_gaze_point_in_user_coordinate_system'][2] -
                                        gdata['left_gaze_origin_in_user_coordinate_system'][2]) ** 2),
                             filepath])


    f.close()


def gaze_data(tracker):
    global global_gaze_data
    tracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)


def start_stop_gaze(event):
    global index_gaze
    index_gaze += 1
    if index_gaze % 2 == 0:
        stop(eyetracker)
    elif index_gaze % 2 == 1:
        gaze_data(eyetracker)

def window_gaze():
    """
    Allows eye tracking with the use of "S" and "Esc" keys
    :return:
    """
    global window
    window = Tk()
    window.attributes('-alpha', 0)
    window.bind("<s>", start_stop_gaze)
    window.bind("<S>", start_stop_gaze)
    window.bind('<Escape>', close)
    window.mainloop()


def stop(tracker):
    global global_gaze_data
    eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)


def close(event):
    stop(eyetracker)
    plot_directory = str(project_directory) + '/Plots'
    os.mkdir(plot_directory)

    df = pd.read_csv(raw_data_name, encoding='ISO-8859-7')  # Makes dataframe df from existing csv

    # Visualization of Gaze By Time before processing data
    global devise_ts
    left_eye_gp_x = df['Left_Gaze_Point_On_Display_Area_X'].tolist()  # Stores left eye X coordinate to list
    right_eye_gp_x = df['Right_Gaze_Point_On_Display_Area_X'].tolist()  # Stores right eye X coordinate to list
    left_eye_gp_y = df['Left_Gaze_Point_On_Display_Area_Y'].tolist()  # Stores left eye Y coordinate to list
    right_eye_gp_y = df['Right_Gaze_Point_On_Display_Area_Y'].tolist()  # Stores right eye Y coordinate to list
    devise_ts = df['Devise_Time'].tolist()  # Stores devise time stamp to list
    date_list = df['Date_time'].tolist()

    left_gp_ucs_x = df['Left_Gaze_Point_In_UCS_X'].tolist()
    left_gp_ucs_y = df['Left_Gaze_Point_In_UCS_Y'].tolist()
    left_gp_ucs_z = df['Left_Gaze_Point_In_UCS_Z'].tolist()
    right_gp_ucs_x = df['Right_Gaze_Point_In_UCS_X'].tolist()
    right_gp_ucs_y = df['Right_Gaze_Point_In_UCS_Y'].tolist()
    right_gp_ucs_z = df['Right_Gaze_Point_In_UCS_Z'].tolist()

    left_go_ucs_x = df['Left_Gaze_Origin_In_UCS_X'].tolist()
    left_go_ucs_y = df['Left_Gaze_Origin_In_UCS_Y'].tolist()
    left_go_ucs_z = df['Left_Gaze_Origin_In_UCS_Z'].tolist()
    right_go_ucs_x = df['Right_Gaze_Origin_In_UCS_X'].tolist()
    right_go_ucs_y = df['Right_Gaze_Origin_In_UCS_Y'].tolist()
    right_go_ucs_z = df['Right_Gaze_Origin_In_UCS_Z'].tolist()

    left_go_tbc_x = df['Left_Gaze_Origin_In_TBC_X'].tolist()
    left_go_tbc_y = df['Left_Gaze_Origin_In_TBC_Y'].tolist()
    left_go_tbc_z = df['Left_Gaze_Origin_In_TBC_Z'].tolist()
    right_go_tbc_x = df['Right_Gaze_Origin_In_TBC_X'].tolist()
    right_go_tbc_y = df['Right_Gaze_Origin_In_TBC_Y'].tolist()
    right_go_tbc_z = df['Right_Gaze_Origin_In_TBC_Z'].tolist()

    right_pupil_diameter = df['Right_Pupil_Diameter'].to_list()
    left_pupil_diameter = df['Left_Pupil_Diameter'].to_list()

    #Turns coordinates into pixels
    try:
        for i in range(len(devise_ts)):
            devise_ts[i] = devise_ts[i] / 1000000
            if math.isnan(left_eye_gp_x[i]) is False:
                left_eye_gp_x[i] = int(left_eye_gp_x[i] * 1920)
            if right_eye_gp_x[i] < 0:
                right_eye_gp_x[i] = 0
            if math.isnan(left_eye_gp_y[i]) is False:
                left_eye_gp_y[i] = int(left_eye_gp_y[i] * 1080)
            if left_eye_gp_y[i] < 0:
                left_eye_gp_y[i] = 0
            if math.isnan(right_eye_gp_y[i]) is False:
                right_eye_gp_y[i] = int(right_eye_gp_y[i] * 1080)
            if right_eye_gp_y[i] < 0:
                right_eye_gp_y[i] = 0
            if math.isnan(right_eye_gp_x[i]) is False:
                right_eye_gp_x[i] = int(right_eye_gp_x[i] * 1920)
            if left_eye_gp_x[i] < 0:
                left_eye_gp_x[i] = 0

        # Plots Left Eye Gaze before Processing
        plotting(devise_ts, left_eye_gp_x, left_eye_gp_y, str(plot_directory) + '/Left_Eye_Before_Processing.png')

        # Plots Right Eye Gaze Data before Processing
        plotting(devise_ts, right_eye_gp_x, right_eye_gp_y, str(plot_directory) + '/Right_Eye_Before_Processing.png')

        # Performs interpolation
        interpolation(left_eye_gp_x)
        interpolation(left_eye_gp_y)
        interpolation(right_eye_gp_x)
        interpolation(right_eye_gp_y)

        interpolation(left_gp_ucs_x)
        interpolation(left_gp_ucs_y)
        interpolation(left_gp_ucs_z)
        interpolation(right_gp_ucs_x)
        interpolation(right_gp_ucs_y)
        interpolation(right_gp_ucs_z)
        interpolation(left_go_ucs_x)
        interpolation(left_go_ucs_y)
        interpolation(left_go_ucs_z)
        interpolation(right_go_ucs_x)
        interpolation(right_go_ucs_y)
        interpolation(right_go_ucs_z)
        interpolation(left_go_tbc_x)
        interpolation(left_go_tbc_y)
        interpolation(left_go_tbc_z)
        interpolation(right_go_tbc_x)
        interpolation(right_go_tbc_y)
        interpolation(right_go_tbc_z)

        interpolation(left_pupil_diameter)
        interpolation(right_pupil_diameter)

        # Gets New Interpolated Values In Dataframe
        df['Left_Gaze_Point_On_Display_Area_X'] = left_eye_gp_x
        df['Right_Gaze_Point_On_Display_Area_X'] = right_eye_gp_x
        df['Left_Gaze_Point_On_Display_Area_Y'] = left_eye_gp_y
        df['Right_Gaze_Point_On_Display_Area_Y'] = right_eye_gp_y

        df['Left_Gaze_Point_In_UCS_X'] = left_gp_ucs_x
        df['Left_Gaze_Point_In_UCS_Y'] = left_gp_ucs_y
        df['Left_Gaze_Point_In_UCS_Z'] = left_gp_ucs_z
        df['Right_Gaze_Point_In_UCS_X'] = right_gp_ucs_x
        df['Right_Gaze_Point_In_UCS_Y'] = right_gp_ucs_y
        df['Right_Gaze_Point_In_UCS_Z'] = right_gp_ucs_z

        df['Left_Gaze_Origin_In_UCS_X'] = left_go_ucs_x
        df['Left_Gaze_Origin_In_UCS_Y'] = left_go_ucs_y
        df['Left_Gaze_Origin_In_UCS_Z'] = left_go_ucs_z
        df['Right_Gaze_Origin_In_UCS_X'] = right_go_ucs_x
        df['Right_Gaze_Origin_In_UCS_Y'] = right_go_ucs_y
        df['Right_Gaze_Origin_In_UCS_Z'] = right_go_ucs_z

        df['Left_Gaze_Origin_In_TBC_X'] = left_go_tbc_x
        df['Left_Gaze_Origin_In_TBC_Y'] = left_go_tbc_y
        df['Left_Gaze_Origin_In_TBC_Z'] = left_go_tbc_z
        df['Right_Gaze_Origin_In_TBC_X'] = right_go_tbc_x
        df['Right_Gaze_Origin_In_TBC_Y'] = right_go_tbc_y
        df['Right_Gaze_Origin_In_TBC_Z'] = right_go_tbc_z

        df['Right_Pupil_Diameter'] = right_pupil_diameter
        df['Left_Pupil_Diameter'] = left_pupil_diameter

        interpolation_data_name = str(project_directory) + '\Interpolated_Eye_Data.csv'
        df.to_csv(interpolation_data_name, index=False)  # Stores Them in an new csv file

        # Plots Left Eye Gaze Point By Time After Interpolation
        plotting(devise_ts, left_eye_gp_x, left_eye_gp_y, str(plot_directory) + '/Left_Eye_After_Interpolation.png')

        # Plots Right Eye Gaze Point By Time After Interpolation
        plotting(devise_ts, right_eye_gp_x, right_eye_gp_y, str(plot_directory) + '/Right_Eye_After_Interpolation.png')

        # Noise Filtering with 4th Order Butterworth Filter

        df = pd.read_csv(interpolation_data_name, encoding='ISO-8859-7')
        # Makes a digital butterworth filter with 5 poles and a cutoff frequency of 5Hz as suggested in the paper
        fc = c.FC  # cut-off frequency
        fs = c.FS  # sampling frequency
        w = fc / (fs / 2)  # normalized frequency
        b, a = signal.butter(c.POLES, w, 'low', analog=False)
        left_eye_gp_x = signal.filtfilt(b, a, left_eye_gp_x)
        left_eye_gp_y = signal.filtfilt(b, a, left_eye_gp_y)
        right_eye_gp_x = signal.filtfilt(b, a, right_eye_gp_x)
        right_eye_gp_y = signal.filtfilt(b, a, right_eye_gp_y)

        left_gp_ucs_x = signal.filtfilt(b, a, left_gp_ucs_x)
        left_gp_ucs_y = signal.filtfilt(b, a, left_gp_ucs_y)
        left_gp_ucs_z = signal.filtfilt(b, a, left_gp_ucs_z)
        right_gp_ucs_x = signal.filtfilt(b, a, right_gp_ucs_x)
        right_gp_ucs_y = signal.filtfilt(b, a, right_gp_ucs_y)
        right_gp_ucs_z = signal.filtfilt(b, a, right_gp_ucs_z)
        left_go_ucs_x = signal.filtfilt(b, a, left_go_ucs_x)
        left_go_ucs_y = signal.filtfilt(b, a, left_go_ucs_y)
        left_go_ucs_z = signal.filtfilt(b, a, left_go_ucs_z)
        right_go_ucs_x = signal.filtfilt(b, a, right_go_ucs_x)
        right_go_ucs_y = signal.filtfilt(b, a, right_go_ucs_y)
        right_go_ucs_z = signal.filtfilt(b, a, right_go_ucs_z)
        left_go_tbc_x = signal.filtfilt(b, a, left_go_tbc_x)
        left_go_tbc_y = signal.filtfilt(b, a, left_go_tbc_y)
        left_go_tbc_z = signal.filtfilt(b, a, left_go_tbc_z)
        right_go_tbc_x = signal.filtfilt(b, a, right_go_tbc_x)
        right_go_tbc_y = signal.filtfilt(b, a, right_go_tbc_y)
        right_go_tbc_z = signal.filtfilt(b, a, right_go_tbc_z)

        for i in range(len(left_eye_gp_x)):
            left_eye_gp_x[i] = int(left_eye_gp_x[i])
            left_eye_gp_y[i] = int(left_eye_gp_y[i])
            right_eye_gp_x[i] = int(right_eye_gp_x[i])
            right_eye_gp_y[i] = int(right_eye_gp_y[i])

        plotting(devise_ts, left_eye_gp_x, left_eye_gp_y, str(plot_directory) + '/Left_Eye_After_Filtering.png')
        plotting(devise_ts, right_eye_gp_x, right_eye_gp_y, str(plot_directory) + '/Right_Eye_After_Filtering.png')

        df['Left_Gaze_Point_On_Display_Area_X'] = left_eye_gp_x  # Gets New Filtered Values In Dataframe
        df['Right_Gaze_Point_On_Display_Area_X'] = right_eye_gp_x
        df['Left_Gaze_Point_On_Display_Area_Y'] = left_eye_gp_y
        df['Right_Gaze_Point_On_Display_Area_Y'] = right_eye_gp_y

        df['Left_Gaze_Point_In_UCS_X'] = left_gp_ucs_x
        df['Left_Gaze_Point_In_UCS_Y'] = left_gp_ucs_y
        df['Left_Gaze_Point_In_UCS_Z'] = left_gp_ucs_z
        df['Right_Gaze_Point_In_UCS_X'] = right_gp_ucs_x
        df['Right_Gaze_Point_In_UCS_Y'] = right_gp_ucs_y
        df['Right_Gaze_Point_In_UCS_Z'] = right_gp_ucs_z

        df['Left_Gaze_Origin_In_UCS_X'] = left_go_ucs_x
        df['Left_Gaze_Origin_In_UCS_Y'] = left_go_ucs_y
        df['Left_Gaze_Origin_In_UCS_Z'] = left_go_ucs_z
        df['Right_Gaze_Origin_In_UCS_X'] = right_go_ucs_x
        df['Right_Gaze_Origin_In_UCS_Y'] = right_go_ucs_y
        df['Right_Gaze_Origin_In_UCS_Z'] = right_go_ucs_z

        df['Left_Gaze_Origin_In_TBC_X'] = left_go_tbc_x
        df['Left_Gaze_Origin_In_TBC_Y'] = left_go_tbc_y
        df['Left_Gaze_Origin_In_TBC_Z'] = left_go_tbc_z
        df['Right_Gaze_Origin_In_TBC_X'] = right_go_tbc_x
        df['Right_Gaze_Origin_In_TBC_Y'] = right_go_tbc_y
        df['Right_Gaze_Origin_In_TBC_Z'] = right_go_tbc_z

        filtered_eye_data = str(project_directory) + '\Filtered_Eye_Data.csv'
        df.to_csv(filtered_eye_data, index=False)  # Stores in csv

        # Typically, at the same times both eyes look at the same fixation point. So, IVT is performed only
        # in left eye
        fixations = ivt(left_eye_gp_x, left_eye_gp_y, devise_ts)

        fixation_count = len(fixations)
        fixation_duration = []

        time1 = []  # time of finish of fixations
        time0 = []  # time of start of fixations
        for k in fixations:
            if len(k) == 1:
                i = k[0]
                t0 = float(devise_ts[i])
                t1 = float(devise_ts[i + 1])
            else:
                i = k[0]
                t0 = float(devise_ts[i])
                t1 = float(devise_ts[k[len(k) - 1] + 1])
            time0.append(t0)
            time1.append(t1)
        time_index = []
        time_findex = []
        for i in range(len(time0)):
            for j in range(len(devise_ts)):
                if time0[i] == float(devise_ts[j]):
                    time_index.append(j)
                if time1[i] == float(devise_ts[j]):
                    time_findex.append(j)
        fixation_duration = []
        saccade_duration = []
        for j in range(len(time0)):
            fixation_duration.append(time1[j] - time0[j])
        for j in range(len(time0) - 1):
            saccade_duration.append(time0[j + 1] - time1[j])

        fixation_x = []
        fixation_y = []
        for i in fixations:
            x = []
            y = []
            for j in range(len(i)):
                x.append(left_eye_gp_x[i[j]])
                y.append(left_eye_gp_y[i[j]])
            fixation_x.append(int(average_list(x)))
            fixation_y.append(int(average_list(y)))

        saccade_amp = []
        for i in range(len(fixation_x) - 1):
            saccade_amp.append(
                np.sqrt((fixation_x[i + 1] - fixation_x[i]) ** 2 + (fixation_y[i + 1] - fixation_y[i]) ** 2))

        fixation_data = str(project_directory) + '\Fixation_Data.csv'
        with open(fixation_data, 'w', newline='') as f:  # Fixation Data are saved in a new csv file
            csv_writer = csv.writer(f)

            csv_writer.writerow(['X Coordinate', 'Y Coordinate', 'Fixation Duration', 'Fixation Start', 'Fixation End'])

            for i in range(len(fixation_x)):
                csv_writer.writerow([fixation_x[i], fixation_y[i], fixation_duration[i], date_list[time_index[i]],
                                     date_list[time_findex[i]]])
            f.close()

        saccade_data = str(project_directory) + '\Saccade_Data.csv'
        with open(saccade_data, 'w', newline='') as f:
            csv_writer = csv.writer(f)

            csv_writer.writerow(['Saccade Amplitude', 'Saccade Duration'])

            for i in range(len(saccade_duration)):
                csv_writer.writerow([saccade_amp[i], saccade_duration[i]])
    except Exception as e:
        pass


    window.destroy()


def appExec2():
    global filtered_eye_data
    buttonBack.place_forget()
    app = QApplication(sys.argv)
    window = Widget()
    window.show()
    app.exec_()
    stop(eyetracker)
    plot_directory = str(project_directory) + '/Plots'
    os.mkdir(plot_directory)

    if opened == 1:
        df = pd.read_csv(raw_data_name, encoding= 'ISO-8859-7')  # Makes dataframe df from existing csv

        # Visualization of Gaze By Time before processing data
        global devise_ts
        left_eye_gp_x = df['Left_Gaze_Point_On_Display_Area_X'].tolist()  # Stores left eye X coordinate to list
        right_eye_gp_x = df['Right_Gaze_Point_On_Display_Area_X'].tolist()  # Stores right eye X coordinate to list
        left_eye_gp_y = df['Left_Gaze_Point_On_Display_Area_Y'].tolist()  # Stores left eye Y coordinate to list
        right_eye_gp_y = df['Right_Gaze_Point_On_Display_Area_Y'].tolist()  # Stores right eye Y coordinate to list
        devise_ts = df['Devise_Time'].tolist()  # Stores devise time stamp to list
        filepaths = df['Filepath'].tolist()

        left_gp_ucs_x = df['Left_Gaze_Point_In_UCS_X'].tolist()
        left_gp_ucs_y = df['Left_Gaze_Point_In_UCS_Y'].tolist()
        left_gp_ucs_z = df['Left_Gaze_Point_In_UCS_Z'].tolist()
        right_gp_ucs_x = df['Right_Gaze_Point_In_UCS_X'].tolist()
        right_gp_ucs_y = df['Right_Gaze_Point_In_UCS_Y'].tolist()
        right_gp_ucs_z = df['Right_Gaze_Point_In_UCS_Z'].tolist()

        left_go_ucs_x = df['Left_Gaze_Origin_In_UCS_X'].tolist()
        left_go_ucs_y = df['Left_Gaze_Origin_In_UCS_Y'].tolist()
        left_go_ucs_z = df['Left_Gaze_Origin_In_UCS_Z'].tolist()
        right_go_ucs_x = df['Right_Gaze_Origin_In_UCS_X'].tolist()
        right_go_ucs_y = df['Right_Gaze_Origin_In_UCS_Y'].tolist()
        right_go_ucs_z = df['Right_Gaze_Origin_In_UCS_Z'].tolist()

        left_go_tbc_x = df['Left_Gaze_Origin_In_TBC_X'].tolist()
        left_go_tbc_y = df['Left_Gaze_Origin_In_TBC_Y'].tolist()
        left_go_tbc_z = df['Left_Gaze_Origin_In_TBC_Z'].tolist()
        right_go_tbc_x = df['Right_Gaze_Origin_In_TBC_X'].tolist()
        right_go_tbc_y = df['Right_Gaze_Origin_In_TBC_Y'].tolist()
        right_go_tbc_z = df['Right_Gaze_Origin_In_TBC_Z'].tolist()

        left_pupil_diameter = df['Left_Pupil_Diameter'].tolist()
        right_pupil_diameter = df['Right_Pupil_Diameter'].tolist()


        # Turns coordinates in pixels and time in seconds
        for i in range(len(devise_ts)):
            devise_ts[i] = devise_ts[i] / 1000000
            if math.isnan(left_eye_gp_x[i]) is False:
                left_eye_gp_x[i] = int(left_eye_gp_x[i] * 1920)
            if math.isnan(left_eye_gp_y[i]) is False:
                left_eye_gp_y[i] = int(left_eye_gp_y[i] * 1080)
            if left_eye_gp_y[i] < 0:
                left_eye_gp_y[i] = 0
            if math.isnan(right_eye_gp_y[i]) is False:
                right_eye_gp_y[i] = int(right_eye_gp_y[i] * 1080)
            if right_eye_gp_y[i] < 0:
                right_eye_gp_y[i] = 0
            if math.isnan(right_eye_gp_x[i]) is False:
                right_eye_gp_x[i] = int(right_eye_gp_x[i] * 1920)
            if left_eye_gp_x[i] < 0:
                left_eye_gp_x[i] = 0
            if right_eye_gp_x[i] < 0:
                right_eye_gp_x[i] = 0



        # Plots Left Eye Gaze before Processing
        plotting(devise_ts, left_eye_gp_x, left_eye_gp_y, str(plot_directory) + '/Left_Eye_Before_Processing.png')

        # Plots Right Eye Gaze Data before Processing
        plotting(devise_ts, right_eye_gp_x, right_eye_gp_y, str(plot_directory) + '/Right_Eye_Before_Processing.png')

        interpolation(left_eye_gp_x)
        interpolation(left_eye_gp_y)
        interpolation(right_eye_gp_x)
        interpolation(right_eye_gp_y)

        interpolation(left_gp_ucs_x)
        interpolation(left_gp_ucs_y)
        interpolation(left_gp_ucs_z)
        interpolation(right_gp_ucs_x)
        interpolation(right_gp_ucs_y)
        interpolation(right_gp_ucs_z)
        interpolation(left_go_ucs_x)
        interpolation(left_go_ucs_y)
        interpolation(left_go_ucs_z)
        interpolation(right_go_ucs_x)
        interpolation(right_go_ucs_y)
        interpolation(right_go_ucs_z)
        interpolation(left_go_tbc_x)
        interpolation(left_go_tbc_y)
        interpolation(left_go_tbc_z)
        interpolation(right_go_tbc_x)
        interpolation(right_go_tbc_y)
        interpolation(right_go_tbc_z)

        interpolation(right_pupil_diameter)
        interpolation(left_pupil_diameter)

        # Gets New Interpolated Values In Dataframe
        df['Left_Gaze_Point_On_Display_Area_X'] = left_eye_gp_x
        df['Right_Gaze_Point_On_Display_Area_X'] = right_eye_gp_x
        df['Left_Gaze_Point_On_Display_Area_Y'] = left_eye_gp_y
        df['Right_Gaze_Point_On_Display_Area_Y'] = right_eye_gp_y

        df['Left_Gaze_Point_In_UCS_X'] = left_gp_ucs_x
        df['Left_Gaze_Point_In_UCS_Y'] = left_gp_ucs_y
        df['Left_Gaze_Point_In_UCS_Z'] = left_gp_ucs_z
        df['Right_Gaze_Point_In_UCS_X'] = right_gp_ucs_x
        df['Right_Gaze_Point_In_UCS_Y'] = right_gp_ucs_y
        df['Right_Gaze_Point_In_UCS_Z'] = right_gp_ucs_z

        df['Left_Gaze_Origin_In_UCS_X'] = left_go_ucs_x
        df['Left_Gaze_Origin_In_UCS_Y'] = left_go_ucs_y
        df['Left_Gaze_Origin_In_UCS_Z'] = left_go_ucs_z
        df['Right_Gaze_Origin_In_UCS_X'] = right_go_ucs_x
        df['Right_Gaze_Origin_In_UCS_Y'] = right_go_ucs_y
        df['Right_Gaze_Origin_In_UCS_Z'] = right_go_ucs_z

        df['Left_Gaze_Origin_In_TBC_X'] = left_go_tbc_x
        df['Left_Gaze_Origin_In_TBC_Y'] = left_go_tbc_y
        df['Left_Gaze_Origin_In_TBC_Z'] = left_go_tbc_z
        df['Right_Gaze_Origin_In_TBC_X'] = right_go_tbc_x
        df['Right_Gaze_Origin_In_TBC_Y'] = right_go_tbc_y
        df['Right_Gaze_Origin_In_TBC_Z'] = right_go_tbc_z

        df['Left_Pupil_Diameter'] = left_pupil_diameter
        df['Right_Pupil_Diameter'] = right_pupil_diameter

        interpolation_data_name = str(project_directory) + '\Interpolated_Eye_Data.csv'
        df.to_csv(interpolation_data_name, index=False, encoding= 'ISO-8859-7')  # Stores Them in an new csv file

        # Plots Left Eye Gaze Point By Time After Interpolation
        plotting(devise_ts, left_eye_gp_x, left_eye_gp_y, str(plot_directory) + '/Left_Eye_After_Interpolation.png')

        # Plots Right Eye Gaze Point By Time After Interpolation
        plotting(devise_ts, right_eye_gp_x, right_eye_gp_y, str(plot_directory) + '/Right_Eye_After_Interpolation.png')

        # Noise Filtering with 4th Order Butterworth Filter

        df = pd.read_csv(interpolation_data_name, encoding='ISO-8859-7')
        # Makes a digital butterworth filter with 5 poles and a cutoff frequency of 5Hz as suggested in the paper
        fc = c.FC  # cut-off frequency
        fs = c.FS  # sampling frequency
        w = fc / (fs / 2)  # normalized frequency
        b, a = signal.butter(c.POLES, w, 'low', analog=False)

        left_eye_gp_x = signal.filtfilt(b, a, left_eye_gp_x)
        left_eye_gp_y = signal.filtfilt(b, a, left_eye_gp_y)
        right_eye_gp_x = signal.filtfilt(b, a, right_eye_gp_x)
        right_eye_gp_y = signal.filtfilt(b, a, right_eye_gp_y)

        left_gp_ucs_x = signal.filtfilt(b, a, left_gp_ucs_x)
        left_gp_ucs_y = signal.filtfilt(b, a, left_gp_ucs_y)
        left_gp_ucs_z = signal.filtfilt(b, a, left_gp_ucs_z)
        right_gp_ucs_x = signal.filtfilt(b, a, right_gp_ucs_x)
        right_gp_ucs_y = signal.filtfilt(b, a, right_gp_ucs_y)
        right_gp_ucs_z = signal.filtfilt(b, a, right_gp_ucs_z)
        left_go_ucs_x = signal.filtfilt(b, a, left_go_ucs_x)
        left_go_ucs_y = signal.filtfilt(b, a, left_go_ucs_y)
        left_go_ucs_z = signal.filtfilt(b, a, left_go_ucs_z)
        right_go_ucs_x = signal.filtfilt(b, a, right_go_ucs_x)
        right_go_ucs_y = signal.filtfilt(b, a, right_go_ucs_y)
        right_go_ucs_z = signal.filtfilt(b, a, right_go_ucs_z)
        left_go_tbc_x = signal.filtfilt(b, a, left_go_tbc_x)
        left_go_tbc_y = signal.filtfilt(b, a, left_go_tbc_y)
        left_go_tbc_z = signal.filtfilt(b, a, left_go_tbc_z)
        right_go_tbc_x = signal.filtfilt(b, a, right_go_tbc_x)
        right_go_tbc_y = signal.filtfilt(b, a, right_go_tbc_y)
        right_go_tbc_z = signal.filtfilt(b, a, right_go_tbc_z)

        for i in range(len(left_eye_gp_x)):
            left_eye_gp_x[i] = int(left_eye_gp_x[i])
            left_eye_gp_y[i] = int(left_eye_gp_y[i])
            right_eye_gp_x[i] = int(right_eye_gp_x[i])
            right_eye_gp_y[i] = int(right_eye_gp_y[i])

        plotting(devise_ts, left_eye_gp_x, left_eye_gp_y, str(plot_directory) + '/Left_Eye_After_Filtering.png')
        plotting(devise_ts, right_eye_gp_x, right_eye_gp_y, str(plot_directory) + '/Right_Eye_After_Filtering.png')

        df['Left_Gaze_Point_On_Display_Area_X'] = left_eye_gp_x  # Gets New Filtered Values In Dataframe
        df['Right_Gaze_Point_On_Display_Area_X'] = right_eye_gp_x
        df['Left_Gaze_Point_On_Display_Area_Y'] = left_eye_gp_y
        df['Right_Gaze_Point_On_Display_Area_Y'] = right_eye_gp_y

        df['Left_Gaze_Point_In_UCS_X'] = left_gp_ucs_x
        df['Left_Gaze_Point_In_UCS_Y'] = left_gp_ucs_y
        df['Left_Gaze_Point_In_UCS_Z'] = left_gp_ucs_z
        df['Right_Gaze_Point_In_UCS_X'] = right_gp_ucs_x
        df['Right_Gaze_Point_In_UCS_Y'] = right_gp_ucs_y
        df['Right_Gaze_Point_In_UCS_Z'] = right_gp_ucs_z

        df['Left_Gaze_Origin_In_UCS_X'] = left_go_ucs_x
        df['Left_Gaze_Origin_In_UCS_Y'] = left_go_ucs_y
        df['Left_Gaze_Origin_In_UCS_Z'] = left_go_ucs_z
        df['Right_Gaze_Origin_In_UCS_X'] = right_go_ucs_x
        df['Right_Gaze_Origin_In_UCS_Y'] = right_go_ucs_y
        df['Right_Gaze_Origin_In_UCS_Z'] = right_go_ucs_z

        df['Left_Gaze_Origin_In_TBC_X'] = left_go_tbc_x
        df['Left_Gaze_Origin_In_TBC_Y'] = left_go_tbc_y
        df['Left_Gaze_Origin_In_TBC_Z'] = left_go_tbc_z
        df['Right_Gaze_Origin_In_TBC_X'] = right_go_tbc_x
        df['Right_Gaze_Origin_In_TBC_Y'] = right_go_tbc_y
        df['Right_Gaze_Origin_In_TBC_Z'] = right_go_tbc_z

        filtered_eye_data = str(project_directory) + '\Filtered_Eye_Data.csv'
        df.to_csv(filtered_eye_data, index=False, encoding='ISO-8859-7')  # Stores in csv

        # Splits lists for each file
        list_of_x = []
        list_of_y = []
        list_of_times = []
        files = []

        filepath = filepaths[0]
        list1 = []
        list2 = []
        list3 = []
        files.append(filepaths[0])
        for i in range(len(filepaths)):
            if filepaths[i] == filepath:
                list1.append(left_eye_gp_x[i])
                list2.append(left_eye_gp_y[i])
                list3.append(devise_ts[i])
            else:
                files.append(filepaths[i])
                list_of_x.append(list1)
                list_of_y.append(list2)
                list_of_times.append(list3)
                list2 = []
                list1 = []
                list3 = []
                list1.append(left_eye_gp_x[i])
                list2.append(left_eye_gp_y[i])
                list3.append(devise_ts[i])
                filepath = filepaths[i]
        list_of_x.append(list1)
        list_of_y.append(list2)
        list_of_times.append(list3)

        fixation_data = str(project_directory) + '\Fixation_Data.csv'
        with open(fixation_data, 'w', newline='') as f:  # Fixation Data are saved in a new csv file
            csv_writer = csv.writer(f)

            csv_writer.writerow(['X Coordinate', 'Y Coordinate', 'Fixation Duration', 'Filename'])

        f.close()

        saccade_data = str(project_directory) + '\Saccade_Data.csv'
        with open(saccade_data, 'w', newline='') as f:
            csv_writer = csv.writer(f)

            csv_writer.writerow(['Saccade Amplitude', 'Saccade Duration', 'Filename'])

        f.close()
        # Typically, at the same times both eyes look at the same fixation point. So, IVT is performed only
        # in left eye
        for r in range(len(list_of_x)):
            fixations = ivt(list_of_x[r], list_of_y[r], list_of_times[r])

            fixation_count = len(fixations)
            fixation_duration = []

            time1 = []  # time of finish of fixations
            time0 = []  # time of start of fixations
            for k in fixations:
                if len(k) == 1:
                    i = k[0]
                    t0 = float(list_of_times[r][i])
                    t1 = float(list_of_times[r][i + 1])
                else:
                    i = k[0]
                    t0 = float(list_of_times[r][i])
                    t1 = float(list_of_times[r][k[len(k) - 1] + 1])
                time0.append(t0)
                time1.append(t1)

            fixation_duration = []
            saccade_duration = []
            for j in range(len(time0)):
                fixation_duration.append(time1[j] - time0[j])
            for j in range(len(time0) - 1):
                saccade_duration.append(time0[j + 1] - time1[j])

            fixation_x = []
            fixation_y = []
            for i in fixations:
                x = []
                y = []
                for j in range(len(i)):
                    x.append(list_of_x[r][i[j]])  # left_eye_gp_x[i[j]])
                    y.append(list_of_y[r][i[j]])
                fixation_x.append(int(average_list(x)))
                fixation_y.append(int(average_list(y)))

            saccade_amp = []
            for i in range(len(fixation_x) - 1):
                saccade_amp.append(
                    np.sqrt((fixation_x[i + 1] - fixation_x[i]) ** 2 + (fixation_y[i + 1] - fixation_y[i]) ** 2))

            with open(fixation_data, 'a', newline='', encoding='ISO-8859-7') as f:  # Fixation Data are saved in a new csv file
                csv_writer = csv.writer(f)

                for i in range(len(fixation_x)):
                    csv_writer.writerow([fixation_x[i], fixation_y[i], fixation_duration[i], files[r]])
            f.close()

            saccade_data = str(project_directory) + '\Saccade_Data.csv'
            with open(saccade_data, 'a', newline='', encoding='ISO-8859-7') as f:
                csv_writer = csv.writer(f)

                for i in range(len(saccade_duration)):
                    csv_writer.writerow([saccade_amp[i], saccade_duration[i], files[r]])
            f.close()

        mid_frame.place(relx=0.5, rely=0.2, relwidth=0.75, relheight=0.25, anchor='n')

        buttonY2.place(relx=0.4, rely=0, relwidth=0.2, relheight=0.4)
        buttonN2.place(relx=0.4, rely=0.5, relwidth=0.2, relheight=0.4)

        outly1.place(relx=0.1, rely=0.1, relwidth=0.1, relheight=0.4, anchor='n')
        outly2.place(relx=0.85, rely=0.1, relwidth=0.1, relheight=0.4)
        outly3.place(relx=0.05, rely=0.45, relwidth=0.9, relheight=0.1)

        window.destroy()

    else:
        end()


def appExec():
    global filtered_eye_data
    buttonBack.place_forget()
    app = QApplication(sys.argv)
    window = Window()
    window.showMaximized()
    app.exec_()
    stop(eyetracker)
    plot_directory = str(project_directory) + '/Plots'
    os.mkdir(plot_directory)

    if opened == 1:
        df = pd.read_csv(raw_data_name, encoding= 'ISO-8859-7')  # Makes dataframe df from existing csv

        # Visualization of Gaze By Time before processing data
        global devise_ts
        left_eye_gp_x = df['Left_Gaze_Point_On_Display_Area_X'].tolist()  # Stores left eye X coordinate to list
        right_eye_gp_x = df['Right_Gaze_Point_On_Display_Area_X'].tolist()  # Stores right eye X coordinate to list
        left_eye_gp_y = df['Left_Gaze_Point_On_Display_Area_Y'].tolist()  # Stores left eye Y coordinate to list
        right_eye_gp_y = df['Right_Gaze_Point_On_Display_Area_Y'].tolist()  # Stores right eye Y coordinate to list
        devise_ts = df['Devise_Time'].tolist()  # Stores devise time stamp to list
        filepaths = df['Filepath'].tolist()

        left_gp_ucs_x = df['Left_Gaze_Point_In_UCS_X'].tolist()
        left_gp_ucs_y = df['Left_Gaze_Point_In_UCS_Y'].tolist()
        left_gp_ucs_z = df['Left_Gaze_Point_In_UCS_Z'].tolist()
        right_gp_ucs_x = df['Right_Gaze_Point_In_UCS_X'].tolist()
        right_gp_ucs_y = df['Right_Gaze_Point_In_UCS_Y'].tolist()
        right_gp_ucs_z = df['Right_Gaze_Point_In_UCS_Z'].tolist()

        left_go_ucs_x = df['Left_Gaze_Origin_In_UCS_X'].tolist()
        left_go_ucs_y = df['Left_Gaze_Origin_In_UCS_Y'].tolist()
        left_go_ucs_z = df['Left_Gaze_Origin_In_UCS_Z'].tolist()
        right_go_ucs_x = df['Right_Gaze_Origin_In_UCS_X'].tolist()
        right_go_ucs_y = df['Right_Gaze_Origin_In_UCS_Y'].tolist()
        right_go_ucs_z = df['Right_Gaze_Origin_In_UCS_Z'].tolist()

        left_go_tbc_x = df['Left_Gaze_Origin_In_TBC_X'].tolist()
        left_go_tbc_y = df['Left_Gaze_Origin_In_TBC_Y'].tolist()
        left_go_tbc_z = df['Left_Gaze_Origin_In_TBC_Z'].tolist()
        right_go_tbc_x = df['Right_Gaze_Origin_In_TBC_X'].tolist()
        right_go_tbc_y = df['Right_Gaze_Origin_In_TBC_Y'].tolist()
        right_go_tbc_z = df['Right_Gaze_Origin_In_TBC_Z'].tolist()

        left_pupil_diameter = df['Left_Pupil_Diameter'].tolist()
        right_pupil_diameter = df['Right_Pupil_Diameter'].tolist()


        # Turns coordinates in pixels and time in seconds
        for i in range(len(devise_ts)):
            devise_ts[i] = devise_ts[i] / 1000000
            if math.isnan(left_eye_gp_x[i]) is False:
                left_eye_gp_x[i] = int(left_eye_gp_x[i] * 1920)
            if math.isnan(left_eye_gp_y[i]) is False:
                left_eye_gp_y[i] = int(left_eye_gp_y[i] * 1080)
            if left_eye_gp_y[i] < 0:
                left_eye_gp_y[i] = 0
            if math.isnan(right_eye_gp_y[i]) is False:
                right_eye_gp_y[i] = int(right_eye_gp_y[i] * 1080)
            if right_eye_gp_y[i] < 0:
                right_eye_gp_y[i] = 0
            if math.isnan(right_eye_gp_x[i]) is False:
                right_eye_gp_x[i] = int(right_eye_gp_x[i] * 1920)
            if left_eye_gp_x[i] < 0:
                left_eye_gp_x[i] = 0
            if right_eye_gp_x[i] < 0:
                right_eye_gp_x[i] = 0



        # Plots Left Eye Gaze before Processing
        plotting(devise_ts, left_eye_gp_x, left_eye_gp_y, str(plot_directory) + '/Left_Eye_Before_Processing.png')

        # Plots Right Eye Gaze Data before Processing
        plotting(devise_ts, right_eye_gp_x, right_eye_gp_y, str(plot_directory) + '/Right_Eye_Before_Processing.png')

        interpolation(left_eye_gp_x)
        interpolation(left_eye_gp_y)
        interpolation(right_eye_gp_x)
        interpolation(right_eye_gp_y)

        interpolation(left_gp_ucs_x)
        interpolation(left_gp_ucs_y)
        interpolation(left_gp_ucs_z)
        interpolation(right_gp_ucs_x)
        interpolation(right_gp_ucs_y)
        interpolation(right_gp_ucs_z)
        interpolation(left_go_ucs_x)
        interpolation(left_go_ucs_y)
        interpolation(left_go_ucs_z)
        interpolation(right_go_ucs_x)
        interpolation(right_go_ucs_y)
        interpolation(right_go_ucs_z)
        interpolation(left_go_tbc_x)
        interpolation(left_go_tbc_y)
        interpolation(left_go_tbc_z)
        interpolation(right_go_tbc_x)
        interpolation(right_go_tbc_y)
        interpolation(right_go_tbc_z)

        interpolation(right_pupil_diameter)
        interpolation(left_pupil_diameter)

        # Gets New Interpolated Values In Dataframe
        df['Left_Gaze_Point_On_Display_Area_X'] = left_eye_gp_x
        df['Right_Gaze_Point_On_Display_Area_X'] = right_eye_gp_x
        df['Left_Gaze_Point_On_Display_Area_Y'] = left_eye_gp_y
        df['Right_Gaze_Point_On_Display_Area_Y'] = right_eye_gp_y

        df['Left_Gaze_Point_In_UCS_X'] = left_gp_ucs_x
        df['Left_Gaze_Point_In_UCS_Y'] = left_gp_ucs_y
        df['Left_Gaze_Point_In_UCS_Z'] = left_gp_ucs_z
        df['Right_Gaze_Point_In_UCS_X'] = right_gp_ucs_x
        df['Right_Gaze_Point_In_UCS_Y'] = right_gp_ucs_y
        df['Right_Gaze_Point_In_UCS_Z'] = right_gp_ucs_z

        df['Left_Gaze_Origin_In_UCS_X'] = left_go_ucs_x
        df['Left_Gaze_Origin_In_UCS_Y'] = left_go_ucs_y
        df['Left_Gaze_Origin_In_UCS_Z'] = left_go_ucs_z
        df['Right_Gaze_Origin_In_UCS_X'] = right_go_ucs_x
        df['Right_Gaze_Origin_In_UCS_Y'] = right_go_ucs_y
        df['Right_Gaze_Origin_In_UCS_Z'] = right_go_ucs_z

        df['Left_Gaze_Origin_In_TBC_X'] = left_go_tbc_x
        df['Left_Gaze_Origin_In_TBC_Y'] = left_go_tbc_y
        df['Left_Gaze_Origin_In_TBC_Z'] = left_go_tbc_z
        df['Right_Gaze_Origin_In_TBC_X'] = right_go_tbc_x
        df['Right_Gaze_Origin_In_TBC_Y'] = right_go_tbc_y
        df['Right_Gaze_Origin_In_TBC_Z'] = right_go_tbc_z

        df['Left_Pupil_Diameter'] = left_pupil_diameter
        df['Right_Pupil_Diameter'] = right_pupil_diameter

        interpolation_data_name = str(project_directory) + '\Interpolated_Eye_Data.csv'
        df.to_csv(interpolation_data_name, index=False, encoding= 'ISO-8859-7')  # Stores Them in an new csv file

        # Plots Left Eye Gaze Point By Time After Interpolation
        plotting(devise_ts, left_eye_gp_x, left_eye_gp_y, str(plot_directory) + '/Left_Eye_After_Interpolation.png')

        # Plots Right Eye Gaze Point By Time After Interpolation
        plotting(devise_ts, right_eye_gp_x, right_eye_gp_y, str(plot_directory) + '/Right_Eye_After_Interpolation.png')

        # Noise Filtering with 4th Order Butterworth Filter

        df = pd.read_csv(interpolation_data_name, encoding='ISO-8859-7')
        # Makes a digital butterworth filter with 5 poles and a cutoff frequency of 5Hz as suggested in the paper
        fc = c.FC  # cut-off frequency
        fs = c.FS  # sampling frequency
        w = fc / (fs / 2)  # normalized frequency
        b, a = signal.butter(c.POLES, w, 'low', analog=False)

        left_eye_gp_x = signal.filtfilt(b, a, left_eye_gp_x)
        left_eye_gp_y = signal.filtfilt(b, a, left_eye_gp_y)
        right_eye_gp_x = signal.filtfilt(b, a, right_eye_gp_x)
        right_eye_gp_y = signal.filtfilt(b, a, right_eye_gp_y)

        left_gp_ucs_x = signal.filtfilt(b, a, left_gp_ucs_x)
        left_gp_ucs_y = signal.filtfilt(b, a, left_gp_ucs_y)
        left_gp_ucs_z = signal.filtfilt(b, a, left_gp_ucs_z)
        right_gp_ucs_x = signal.filtfilt(b, a, right_gp_ucs_x)
        right_gp_ucs_y = signal.filtfilt(b, a, right_gp_ucs_y)
        right_gp_ucs_z = signal.filtfilt(b, a, right_gp_ucs_z)
        left_go_ucs_x = signal.filtfilt(b, a, left_go_ucs_x)
        left_go_ucs_y = signal.filtfilt(b, a, left_go_ucs_y)
        left_go_ucs_z = signal.filtfilt(b, a, left_go_ucs_z)
        right_go_ucs_x = signal.filtfilt(b, a, right_go_ucs_x)
        right_go_ucs_y = signal.filtfilt(b, a, right_go_ucs_y)
        right_go_ucs_z = signal.filtfilt(b, a, right_go_ucs_z)
        left_go_tbc_x = signal.filtfilt(b, a, left_go_tbc_x)
        left_go_tbc_y = signal.filtfilt(b, a, left_go_tbc_y)
        left_go_tbc_z = signal.filtfilt(b, a, left_go_tbc_z)
        right_go_tbc_x = signal.filtfilt(b, a, right_go_tbc_x)
        right_go_tbc_y = signal.filtfilt(b, a, right_go_tbc_y)
        right_go_tbc_z = signal.filtfilt(b, a, right_go_tbc_z)

        for i in range(len(left_eye_gp_x)):
            left_eye_gp_x[i] = int(left_eye_gp_x[i])
            left_eye_gp_y[i] = int(left_eye_gp_y[i])
            right_eye_gp_x[i] = int(right_eye_gp_x[i])
            right_eye_gp_y[i] = int(right_eye_gp_y[i])

        plotting(devise_ts, left_eye_gp_x, left_eye_gp_y, str(plot_directory) + '/Left_Eye_After_Filtering.png')
        plotting(devise_ts, right_eye_gp_x, right_eye_gp_y, str(plot_directory) + '/Right_Eye_After_Filtering.png')

        df['Left_Gaze_Point_On_Display_Area_X'] = left_eye_gp_x  # Gets New Filtered Values In Dataframe
        df['Right_Gaze_Point_On_Display_Area_X'] = right_eye_gp_x
        df['Left_Gaze_Point_On_Display_Area_Y'] = left_eye_gp_y
        df['Right_Gaze_Point_On_Display_Area_Y'] = right_eye_gp_y

        df['Left_Gaze_Point_In_UCS_X'] = left_gp_ucs_x
        df['Left_Gaze_Point_In_UCS_Y'] = left_gp_ucs_y
        df['Left_Gaze_Point_In_UCS_Z'] = left_gp_ucs_z
        df['Right_Gaze_Point_In_UCS_X'] = right_gp_ucs_x
        df['Right_Gaze_Point_In_UCS_Y'] = right_gp_ucs_y
        df['Right_Gaze_Point_In_UCS_Z'] = right_gp_ucs_z

        df['Left_Gaze_Origin_In_UCS_X'] = left_go_ucs_x
        df['Left_Gaze_Origin_In_UCS_Y'] = left_go_ucs_y
        df['Left_Gaze_Origin_In_UCS_Z'] = left_go_ucs_z
        df['Right_Gaze_Origin_In_UCS_X'] = right_go_ucs_x
        df['Right_Gaze_Origin_In_UCS_Y'] = right_go_ucs_y
        df['Right_Gaze_Origin_In_UCS_Z'] = right_go_ucs_z

        df['Left_Gaze_Origin_In_TBC_X'] = left_go_tbc_x
        df['Left_Gaze_Origin_In_TBC_Y'] = left_go_tbc_y
        df['Left_Gaze_Origin_In_TBC_Z'] = left_go_tbc_z
        df['Right_Gaze_Origin_In_TBC_X'] = right_go_tbc_x
        df['Right_Gaze_Origin_In_TBC_Y'] = right_go_tbc_y
        df['Right_Gaze_Origin_In_TBC_Z'] = right_go_tbc_z

        filtered_eye_data = str(project_directory) + '\Filtered_Eye_Data.csv'
        df.to_csv(filtered_eye_data, index=False, encoding='ISO-8859-7')  # Stores in csv

        # Splits lists for each file
        list_of_x = []
        list_of_y = []
        list_of_times = []
        files = []

        filepath = filepaths[0]
        list1 = []
        list2 = []
        list3 = []
        files.append(filepaths[0])
        for i in range(len(filepaths)):
            if filepaths[i] == filepath:
                list1.append(left_eye_gp_x[i])
                list2.append(left_eye_gp_y[i])
                list3.append(devise_ts[i])
            else:
                files.append(filepaths[i])
                list_of_x.append(list1)
                list_of_y.append(list2)
                list_of_times.append(list3)
                list2 = []
                list1 = []
                list3 = []
                list1.append(left_eye_gp_x[i])
                list2.append(left_eye_gp_y[i])
                list3.append(devise_ts[i])
                filepath = filepaths[i]
        list_of_x.append(list1)
        list_of_y.append(list2)
        list_of_times.append(list3)

        fixation_data = str(project_directory) + '\Fixation_Data.csv'
        with open(fixation_data, 'w', newline='') as f:  # Fixation Data are saved in a new csv file
            csv_writer = csv.writer(f)

            csv_writer.writerow(['X Coordinate', 'Y Coordinate', 'Fixation Duration', 'Filename'])

        f.close()

        saccade_data = str(project_directory) + '\Saccade_Data.csv'
        with open(saccade_data, 'w', newline='') as f:
            csv_writer = csv.writer(f)

            csv_writer.writerow(['Saccade Amplitude', 'Saccade Duration', 'Filename'])

        f.close()
        # Typically, at the same times both eyes look at the same fixation point. So, IVT is performed only
        # in left eye
        for r in range(len(list_of_x)):
            fixations = ivt(list_of_x[r], list_of_y[r], list_of_times[r])

            fixation_count = len(fixations)
            fixation_duration = []

            time1 = []  # time of finish of fixations
            time0 = []  # time of start of fixations
            for k in fixations:
                if len(k) == 1:
                    i = k[0]
                    t0 = float(list_of_times[r][i])
                    t1 = float(list_of_times[r][i + 1])
                else:
                    i = k[0]
                    t0 = float(list_of_times[r][i])
                    t1 = float(list_of_times[r][k[len(k) - 1] + 1])
                time0.append(t0)
                time1.append(t1)

            fixation_duration = []
            saccade_duration = []
            for j in range(len(time0)):
                fixation_duration.append(time1[j] - time0[j])
            for j in range(len(time0) - 1):
                saccade_duration.append(time0[j + 1] - time1[j])

            fixation_x = []
            fixation_y = []
            for i in fixations:
                x = []
                y = []
                for j in range(len(i)):
                    x.append(list_of_x[r][i[j]])  # left_eye_gp_x[i[j]])
                    y.append(list_of_y[r][i[j]])
                fixation_x.append(int(average_list(x)))
                fixation_y.append(int(average_list(y)))

            saccade_amp = []
            for i in range(len(fixation_x) - 1):
                saccade_amp.append(
                    np.sqrt((fixation_x[i + 1] - fixation_x[i]) ** 2 + (fixation_y[i + 1] - fixation_y[i]) ** 2))

            with open(fixation_data, 'a', newline='', encoding='ISO-8859-7') as f:  # Fixation Data are saved in a new csv file
                csv_writer = csv.writer(f)

                for i in range(len(fixation_x)):
                    csv_writer.writerow([fixation_x[i], fixation_y[i], fixation_duration[i], files[r]])
            f.close()

            saccade_data = str(project_directory) + '\Saccade_Data.csv'
            with open(saccade_data, 'a', newline='', encoding='ISO-8859-7') as f:
                csv_writer = csv.writer(f)

                for i in range(len(saccade_duration)):
                    csv_writer.writerow([saccade_amp[i], saccade_duration[i], files[r]])
            f.close()
        global fixation_dir
        global scanpath_dir
        fixation_dir = str(project_directory) + '/Fixation Maps'
        os.mkdir(fixation_dir)
        scanpath_dir = str(project_directory) + '/Scanpath'
        #os.mkdir(scanpath_dir)
        df = pd.read_csv(fixation_data, encoding='ISO-8859-7')
        filen = df['Filename'].tolist()
        fix_x = df['X Coordinate'].tolist()
        fix_y = df['Y Coordinate'].tolist()
        fix_duration = df['Fixation Duration'].tolist()
        for filename in os.listdir(image_dir):
            image_name = str(image_dir) + '/' + filename
            filex = []
            filey = []
            filedur = []
            for i in range(len(fix_x)):
                if Path(image_name).stem == Path(filen[i]).stem:
                    filex.append(fix_x[i])
                    filey.append(fix_y[i])
                    filedur.append(fix_duration[i])
            FixationScan(filex, filey, filedur, image_name)
            #Scanpath(filex, filey, image_name)

        mid_frame.place(relx=0.5, rely=0.2, relwidth=0.75, relheight=0.25, anchor='n')

        buttonY.place(relx=0.4, rely=0, relwidth=0.2, relheight=0.4)
        buttonN.place(relx=0.4, rely=0.5, relwidth=0.2, relheight=0.4)

        outly1.place(relx=0.1, rely=0.1, relwidth=0.1, relheight=0.4, anchor='n')
        outly2.place(relx=0.85, rely=0.1, relwidth=0.1, relheight=0.4)
        outly3.place(relx=0.05, rely=0.45, relwidth=0.9, relheight=0.1)

        window.destroy()
    else:
        end()


class Widget(QtWidgets.QWidget):
    """
    Video Player
    """
    def __init__(self, parent=None):

        super(Widget, self).__init__(parent)

        # first window,just have a single button for play the video
        self.resize(256, 100)
        p = self.palette()
        p.setColor(QPalette.Window, Qt.white)
        self.setPalette(p)
        self.setWindowIcon(QIcon('eye_icon.png'))

        self.btn_play = QtWidgets.QPushButton(self)
        self.btn_play.setGeometry(QtCore.QRect(20, 20, 220, 70))
        self.btn_play.setText("Browse File")
        self.btn_play.setStyleSheet("background-color:'#D1EEEE';")
        self.btn_play.clicked.connect(self.Play_video)  # click to play video


        self._scene = QtWidgets.QGraphicsScene(self)
        self._gv = QtWidgets.QGraphicsView(self._scene)
        self._gv.setBackgroundBrush(Qt.black)
        self._gv.setWindowIcon(QIcon('eye_icon.png'))
        # construct a videoitem for showing the video
        self._videoitem = QtMultimediaWidgets.QGraphicsVideoItem()
        # add it into the scene
        self._scene.addItem(self._videoitem)
        self.filename = ''

        # self._scene.addItem(self._ellipse_item)
        self._gv.fitInView(self._videoitem)

        self._player = QtMultimedia.QMediaPlayer(
            self, QtMultimedia.QMediaPlayer.VideoSurface
        )
        self._player.setVideoOutput(self._videoitem)
        self._player.setMedia(QMediaContent(QUrl.fromLocalFile(self.filename)))
        QShortcut(
            QKeySequence(Qt.Key_Space),
            self._gv,
            self.get_coords
        )
        self._player.mediaStatusChanged.connect(self.statusChanged)


    def Play_video(self):
        global filepath
        global opened
        self.filename, _ = QFileDialog.getOpenFileName(self, "Browse File", '', 'Video (*.mp4 *.flv *.mov *.wmv *.avi '
                                                                           '*.mkv )')
        while len(self.filename) == 0:
            self.filename, _ = QFileDialog.getOpenFileName(self, "Browse File", '',
                                                           'Video (*.mp4 *.flv *.mov *.wmv *.avi '
                                                           '*.mkv )')
        filepath = self.filename

        self._player.setMedia(QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(self.filename)))

        size = QtCore.QSizeF(1600, 960)  # I hope it can fullscreen the video
        self._videoitem.setSize(size)
        self._gv.resize(1888, 1062)
        self._gv.showMaximized()

        self.hide()
        opened = 1
        gaze_data(eyetracker)
        self._player.play()
        self.hide()

    def statusChanged(self, status):
        if status == QMediaPlayer.EndOfMedia:
            stop(eyetracker)

    def get_coords(self):
        """
        Saves cursor coordinates to csv
        :return:
        """
        x,y = pyautogui.position()
        with open(list_name, 'a', newline='') as p:
            csv_writer = csv.writer(p)
            csv_writer.writerow([x, y])
        p.close()


# Images Window
class Window(QWidget):
    """
    Image Player
    """
    def __init__(self):
        super().__init__()

        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()

        self.setWindowTitle("Media Player")
        self.setGeometry(0, 0, user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))
        self.setWindowIcon(QIcon('eye_icon.png'))

        p = self.palette()
        p.setColor(QPalette.Window, Qt.black)
        self.setPalette(p)

        self.init_ui()

    def init_ui(self):
        # Create a Media Player Object
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        self.mediaPlayer.mediaStatusChanged.connect(self.statusChanged)

        # Create a Video Widget Object
        videowidget = QVideoWidget()
        # Create Open Button
        openBtn = QPushButton('Browse File')
        openBtn.clicked.connect(stop)
        openBtn.clicked.connect(self.open_file)

        # Create Play Button
        self.playBtn = QPushButton()
        self.playBtn.setEnabled(False)
        self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playBtn.clicked.connect(self.play_video)
        self.playBtn.clicked.connect(self.screen)

        # Create Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.set_position)

        # Create Label
        self.label = QLabel()
        self.label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        # Create Hbox Layout
        hboxLayout = QHBoxLayout()
        hboxLayout.setContentsMargins(0, 0, 0, 0)

        # Set Widgets to the Hbox layout
        hboxLayout.addWidget(openBtn)
        hboxLayout.addWidget(self.playBtn)
        hboxLayout.addWidget(self.slider)

        # Create Vbox Layout
        vboxLayout = QVBoxLayout()
        vboxLayout.addWidget(videowidget)
        vboxLayout.addLayout(hboxLayout)
        vboxLayout.addWidget(self.label)

        self.setLayout(vboxLayout)

        self.mediaPlayer.setVideoOutput(videowidget)

        # Media Player Signals

        self.mediaPlayer.stateChanged.connect(self.mediastate_changed)
        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.durationChanged.connect(self.duration_changed)

        self.hidden = True

    def screen(self):
        """
        Takes and saves Screenshot of shown image
        """
        time.sleep(1)
        file_base = Path(self.filename).stem
        myScreenshot = pyautogui.screenshot()
        name = str(image_dir) + '/' + file_base
        myScreenshot.save(name + '.png')

    def hide_unhide(self):
        if self.hidden:
            self.playBtn.show()
            self.hidden = False
        else:
            self.playBtn.hide()
            self.hidden = True

    def open_file(self):
        self.filename, _ = QFileDialog.getOpenFileName(self, "Browse File", '', 'Images (*.png '
                                                                                '*.jpeg *.jpg *.bmp *.gif)')

        while len(self.filename) == 0:
            self.filename, _ = QFileDialog.getOpenFileName(self, "Browse File", '', 'Images (*.png '
                                                                                '*.jpeg *.jpg *.bmp *.gif)')
        global filepath
        global opened
        filepath = self.filename
        if self.filename != '':
            #opened = 1
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(self.filename)))
            self.playBtn.setEnabled(True)

    def play_video(self):
        global opened
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            if self.filename.endswith('.mp4') or self.filename.endswith('.flv') or self.filename.endswith('.mov') \
                    or self.filename.endswith('.wmv') or self.filename.endswith('.avi') or self.filename.endswith(
                     '.mkv'):
                stop(eyetracker)
                self.mediaPlayer.pause()
            else:
                self.playBtn.setEnabled(False)
        else:
            gaze_data(eyetracker)
            self.mediaPlayer.play()

            if self.filename.endswith('.png') or self.filename.endswith('.jpeg') or self.filename.endswith('.jpg') \
                    or self.filename.endswith('.bmp') or self.filename.endswith('.gif'):
                self.playBtn.setEnabled(False)
                opened = 1

    def mediastate_changed(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playBtn.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPause)
            )
        else:
            self.playBtn.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay)
            )

    def position_changed(self, position):
        self.slider.setValue(position)

    def duration_changed(self, duration):
        self.slider.setRange(0, duration)

    def set_position(self, position):
        self.mediaPlayer.setPosition(position)

    def handle_errors(self):
        self.playBtn.setEnabled(False)
        self.label.setText("Error:" + self.mediaPlayer.errorString())

    def statusChanged(self, status):
        if status == QMediaPlayer.EndOfMedia:
            if self.filename.endswith('.mp4') or self.filename.endswith('.flv') or self.filename.endswith('.mov') \
                    or self.filename.endswith('.wmv') or self.filename.endswith('.avi') or self.filename.endswith(
                '.mkv'):
                stop(eyetracker)
            else:
                self.playBtn.setEnabled(False)


# Eye Tracker Manager Call Function
def call_eyetracker_manager(address, mode):
    """
    Calls Eye Tracker Manager for display area settings and calibration.
    """
    global home_directory
    try:
        os_type = platform.system()
        etm_path = ''
        if os_type == "Windows":
            # print("Operating System: Windows")
            etm_path = home_directory + \
                       "/AppData/Local/Programs/TobiiProEyeTrackerManager/TobiiProEyeTrackerManager.exe"
        elif os_type == "Linux":
            print("Operating System: Linux")
            etm_path = "TobiiProEyeTrackerManager"
        elif os_type == "Darwin":
            etm_path = \
                "/Applications/TobiiProEyeTrackerManager.app/Contents/MacOS/TobiiProEyeTrackerManager"
        else:
            print("Unsupported...")
            exit(1)
        # Opens Eye Tracker Manager in calibration mode
        _ = subprocess.check_output(
            [etm_path])

    except Exception as e:
        print(e)


def second_q():
    buttonBack3.place_forget()
    buttonBack6.place(relx=0, relwidth=0.2, relheight=1)
    label.config(text="Επιλέξτε")
    buttonYes.place_forget()
    buttonNo.place_forget()
    mid_frame.place(relx=0.5, rely=0.2, relwidth=0.75, relheight=0.6, anchor='n')
    outly2.place(relx=0.85, rely=0.1, relwidth=0.1, relheight=0.8)
    outly3.place(relx=0.05, rely=0.8, relwidth=0.9, relheight=0.1)
    outly1.place(relx=0.1, rely=0.1, relwidth=0.1, relheight=0.8, anchor='n')


    buttonDisplay.place(relx=0.2, relwidth=0.6, rely=0, relheight=0.2)
    buttonDA.place(relx=0.2, relwidth=0.6, rely=0.2, relheight=0.2)
    buttonGaze.place(relx=0.2, relwidth=0.6, rely=0.4, relheight=0.2)
    buttonExp.place(relx=0.2, relwidth=0.6, rely=0.6, relheight=0.2)
    buttonVid.place(relx=0.2, relwidth=0.6, rely=0.8, relheight=0.2)


def dip_properties():
    label.config(text='Tobii Pro Nano:')
    buttonBack6.place_forget()
    appear_frame.config(bg='black')
    appear_frame.place(relx=0.5, rely=0.2, relwidth=0.7, relheight=0.6, anchor='n')
    label3[
        'text'] = "\nΔιεύθυνση: " + eyetracker.address + "\nΜοντέλο: " + eyetracker.model + "\nΌνομα: " + eyetracker.device_name + "\nΣειριακός Αριθμός:  " + eyetracker.serial_number
    label3.config(font=('Helvetica bold', 14))
    label3.place(relwidth=1, relheight=1)
    buttonBack.place(relx=0, relwidth=0.2, relheight=1)


def back():
    entry.place_forget()
    label4.place_forget()
    buttonSave.place_forget()
    label.config(text="Επιλέξτε")
    buttonBack.place_forget()
    buttonBack6.place(relx=0, relwidth=0.2, relheight=1)
    stop(eyetracker)
    appear_frame.place_forget()
    label3.place_forget()


def get_name():
    home_directory = os.path.expanduser("~")  # Finds home directory (cross platform)
    global raw_data_name
    global filepath
    global project_directory
    filepath = ''
    global entry
    project_name = entry.get()
    while project_name[-1] == ' ':
        project_name = project_name[:-1]
    project_directory = os.path.join(home_directory, project_name)
    if os.path.exists(project_directory):
        label4['text'] = "Το αρχείο υπάρχει ήδη. Προσπαθήστε ξανά"
    else:
        entry.place_forget()
        label4.place_forget()
        buttonSave.place_forget()
        appear_frame.place_forget()
        label5 = tk.Label(root, text="Thanks For Joining!", bg='#80c1ff')
        label5.config(font=40)
        label5.place(relwidth=1, relheight=1)
        os.mkdir(project_directory)
        raw_data_name = str(project_directory) + '\Raw_Data.csv'  # prepei na to do analoga
        global list_name
        list_name = str(project_directory) + '\List.csv'
        with open(list_name, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['X', 'Y'])
        with open(raw_data_name, 'w', newline='') as f:  # Data are saved in a new csv file
            csv_writer = csv.writer(f)

            # Writes the names of the Headers in first line
            csv_writer.writerow(
                ['Devise_Time', 'Left_Gaze_Point_On_Display_Area_X', 'Right_Gaze_Point_On_Display_Area_X',
                 'Left_Gaze_Point_On_Display_Area_Y', 'Right_Gaze_Point_On_Display_Area_Y',
                 'Left_Gaze_Point_In_UCS_X', 'Left_Gaze_Point_In_UCS_Y', 'Left_Gaze_Point_In_UCS_Z',
                 'Right_Gaze_Point_In_UCS_X', 'Right_Gaze_Point_In_UCS_Y', 'Right_Gaze_Point_In_UCS_Z',
                 'Left_Gaze_Point_Validity', 'Right_Gaze_Point_Validity', 'Left_Pupil_Diameter',
                 'Right_Pupil_Diameter', 'Left_Pupil_Validity', 'Right_Pupil_Validity',
                 'Left_Gaze_Origin_In_UCS_X', 'Left_Gaze_Origin_In_UCS_Y', 'Left_Gaze_Origin_In_UCS_Z',
                 'Right_Gaze_Origin_In_UCS_X', 'Right_Gaze_Origin_In_UCS_Y', 'Right_Gaze_Origin_In_UCS_Z',
                 'Left_Gaze_Origin_In_TBC_X', 'Left_Gaze_Origin_In_TBC_Y', 'Left_Gaze_Origin_In_TBC_Z',
                 'Right_Gaze_Origin_In_TBC_X', 'Right_Gaze_Origin_In_TBC_Y', 'Right_Gaze_Origin_In_TBC_Z',
                 'Left_Gaze_Origin_Validity', 'Right_Gaze_Origin_Validity', 'System_Time', 'Date_time',
                 'Gaze_Vector_Distance', 'Filepath'])
        f.close()
        root.destroy()
        window_gaze()


def start_progress():
    threading.Thread(target=get_map).start()


def start_progress2():
    threading.Thread(target=get_map_Im).start()


def end():
    mid_frame.place_forget()
    frame2.place(relx=0, rely=0.08, relwidth=1, relheight=0.34)
    frame.place(relx=0.5, rely=0.1, relwidth=1, relheight=0.3)
    outly1.place_forget()
    outly2.place_forget()
    outly3.place_forget()
    outly6.place(relx=0.89, rely=0, relwidth=0.2, relheight=1)
    outly2.place(relx=0.9, rely=0, relwidth=0.05, relheight=1)
    outly3.place_forget()
    outly4.place(relx=0.975, rely=0, relwidth=0.01, relheight=1)
    outly5.place(relx=0.958, rely=0, relwidth=0.01, relheight=1)
    label.config(text="Τέλος πειράματος")
    label.config(font=('Helvetica bold', 48))


def get_map():
    global opened
    global end_map
    end_map = 0
    buttonY.place_forget()
    buttonN.place_forget()
    label['text'] = 'Η διαδικασία μπορεί να διαρκέσει λίγα λεπτά...'
    progress.place(relx=0.1, rely=0.5, relwidth=0.8, relheight=0.2)
    filtered_eye_data = str(project_directory) + '\Filtered_Eye_Data.csv'
    df = pd.read_csv(filtered_eye_data, encoding='ISO-8859-7')
    left_eye_gp_x = df['Left_Gaze_Point_On_Display_Area_X'].tolist()
    left_eye_gp_y = df['Left_Gaze_Point_On_Display_Area_Y'].tolist()
    files = df['Filepath'].tolist()
    global heatmap_dir
    heatmap_dir = str(project_directory) + '/Heatmaps'
    os.mkdir(heatmap_dir)
    for filename in os.listdir(image_dir):
        image_name = str(image_dir) + '/' + filename
        img1 = imageio.imread(image_name)
        img = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)

        # Generate toy fixation data
        # when you use, replace here with your data
        H, W, _ = img.shape

        list_of_1 = []
        leftx = []
        lefty = []
        for i in range(len(left_eye_gp_x)):
            if Path(image_name).stem == Path(files[i]).stem:
                list_of_1.append(1)
                leftx.append(left_eye_gp_x[i])
                lefty.append(left_eye_gp_y[i])

        fix_arr = np.array([leftx, lefty, list_of_1])
        fix_arr = fix_arr.transpose()

        heatmap = Fixpos2Densemap(fix_arr, 1920, 1080, img, 0.7, 5)
        filename_path = str(image_dir) + '/' + str(filename)
        file_stem = Path(filename_path).stem

        name = str(heatmap_dir) + '/' + str(file_stem) + '_Heatmap.png'
        is_success, im_buf_arr = cv2.imencode(".png", heatmap)
        im_buf_arr.tofile(name)

        cv2.imwrite(name, heatmap)

    progress.place_forget()
    end_map = 1
    end()


def get_map_Im():
    c1.place_forget()
    #c2.place_forget()
    c3.place_forget()
    buttonImGet.place_forget()

    label['text'] = 'Η διαδικασία μπορεί να διαρκέσει λίγα λεπτά...'
    progress.place(relx=0.1, rely=0.5, relwidth=0.8, relheight=0.2)
    filtered_eye_data = root.filename_csv_fil
    fixation_eye_data = root.filename_csv_fix
    df = pd.read_csv(filtered_eye_data, encoding='ISO-8859-7')
    df1 = pd.read_csv(fixation_eye_data, encoding='ISO-8859-7')
    left_eye_gp_x = df['Left_Gaze_Point_On_Display_Area_X'].tolist()
    left_eye_gp_y = df['Left_Gaze_Point_On_Display_Area_Y'].tolist()
    files_fil = df['Filepath'].to_list()
    date_time = df['Date_time'].tolist()
    img1 = imageio.imread(root.filename_im)
    img = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    H, W, _ = img.shape

    list_of_1 = []
    leftx = []
    lefty = []

    if date_start == None and date_finish == None:
        if 'Fixation Start' not in df1.columns:
            for i in range(len(files_fil)):
                if Path(root.filename_im).stem == Path(files_fil[i]).stem:
                    leftx.append(left_eye_gp_x[i])
                    lefty.append(left_eye_gp_y[i])
                    list_of_1.append(1)
        else:
            for i in range(len(left_eye_gp_x)):
                leftx.append(left_eye_gp_x[i])
                lefty.append(left_eye_gp_y[i])
                list_of_1.append(1)
    else:
        for i in range(len(left_eye_gp_x)):
            if date_time[i] > date_time[index_start] and date_time[i] < date_time[index_stop]:
                list_of_1.append(1)
                leftx.append(left_eye_gp_x[i])
                lefty.append(left_eye_gp_y[i])

    fix_arr = np.array([leftx, lefty, list_of_1])
    fix_arr = fix_arr.transpose()

    heatmap = Fixpos2Densemap(fix_arr, 1920, 1080, img, 0.7, 5)
    name = uniquify(str(home_directory) + '/' + 'Heatmap.png')
    is_success, im_buf_arr = cv2.imencode(".png", heatmap)
    im_buf_arr.tofile(name)
    #cv2.imwrite(name, heatmap)
    progress.place_forget()
    label.config(text='Αρχικό Μενού')
    buttonEye.place(relx=0.4, rely=0, relwidth=0.2, relheight=0.4)
    buttonData.place(relx=0.4, rely=0.5, relwidth=0.2, relheight=0.4)


def get_name3():
    home_directory = os.path.expanduser("~")  # Finds home directory (cross platform)
    global raw_data_name
    global filepath
    #global image_dir
    filepath = ''
    global entry
    project_name = entry.get()
    while project_name[-1] == ' ':
        project_name = project_name[:-1]
    global project_directory
    project_directory = os.path.join(home_directory, project_name)
    if os.path.exists(project_directory):
        label4['text'] = "Το άρχείο υπάρχει ήδη. Προσπαθήστε ξανά"
    else:
        os.mkdir(project_directory)
        entry.place_forget()
        buttonSave.place_forget()
        appear_frame.place_forget()
        raw_data_name = str(project_directory) + '\Raw_Data.csv'
        global list_name
        list_name = str(project_directory) + '\List.csv'
        with open(list_name, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['X', 'Y'])
        f.close()
        with open(raw_data_name, 'w', newline='') as f:  # Data are saved in a new csv file
            csv_writer = csv.writer(f)

            # Writes the names of the Headers in first line
            csv_writer.writerow(
                ['Devise_Time', 'Left_Gaze_Point_On_Display_Area_X', 'Right_Gaze_Point_On_Display_Area_X',
                 'Left_Gaze_Point_On_Display_Area_Y', 'Right_Gaze_Point_On_Display_Area_Y',
                 'Left_Gaze_Point_In_UCS_X', 'Left_Gaze_Point_In_UCS_Y', 'Left_Gaze_Point_In_UCS_Z',
                 'Right_Gaze_Point_In_UCS_X', 'Right_Gaze_Point_In_UCS_Y', 'Right_Gaze_Point_In_UCS_Z',
                 'Left_Gaze_Point_Validity', 'Right_Gaze_Point_Validity', 'Left_Pupil_Diameter',
                 'Right_Pupil_Diameter', 'Left_Pupil_Validity', 'Right_Pupil_Validity',
                 'Left_Gaze_Origin_In_UCS_X', 'Left_Gaze_Origin_In_UCS_Y', 'Left_Gaze_Origin_In_UCS_Z',
                 'Right_Gaze_Origin_In_UCS_X', 'Right_Gaze_Origin_In_UCS_Y', 'Right_Gaze_Origin_In_UCS_Z',
                 'Left_Gaze_Origin_In_TBC_X', 'Left_Gaze_Origin_In_TBC_Y', 'Left_Gaze_Origin_In_TBC_Z',
                 'Right_Gaze_Origin_In_TBC_X', 'Right_Gaze_Origin_In_TBC_Y', 'Right_Gaze_Origin_In_TBC_Z',
                 'Left_Gaze_Origin_Validity', 'Right_Gaze_Origin_Validity', 'System_Time', 'Date_time',
                 'Gaze_Vector_Distance', 'Filepath'])
        f.close()

        buttonVid.place_forget()
        buttonDisplay.place_forget()
        buttonExp.place_forget()
        buttonGaze.place_forget()
        buttonDA.place_forget()

        appExec2()


def get_name2():
    home_directory = os.path.expanduser("~")  # Finds home directory (cross platform)
    global raw_data_name
    global filepath
    global image_dir
    filepath = ''
    global entry
    project_name = entry.get()
    while project_name[-1] == ' ':
        project_name = project_name[:-1]
    global project_directory
    project_directory = os.path.join(home_directory, project_name)
    if os.path.exists(project_directory):
        label4['text'] = "Το άρχείο υπάρχει ήδη. Προσπαθήστε ξανά"
    else:
        os.mkdir(project_directory)
        image_dir = str(project_directory) + '/Images'
        os.mkdir(image_dir)
        entry.place_forget()
        buttonSave.place_forget()
        appear_frame.place_forget()
        raw_data_name = str(project_directory) + '\Raw_Data.csv'
        global list_name
        list_name = str(project_directory) + '\List.csv'
        with open(list_name, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['X', 'Y'])
        f.close()
        with open(raw_data_name, 'w', newline='') as f:  # Data are saved in a new csv file
            csv_writer = csv.writer(f)

            # Writes the names of the Headers in first line
            csv_writer.writerow(
                ['Devise_Time', 'Left_Gaze_Point_On_Display_Area_X', 'Right_Gaze_Point_On_Display_Area_X',
                 'Left_Gaze_Point_On_Display_Area_Y', 'Right_Gaze_Point_On_Display_Area_Y',
                 'Left_Gaze_Point_In_UCS_X', 'Left_Gaze_Point_In_UCS_Y', 'Left_Gaze_Point_In_UCS_Z',
                 'Right_Gaze_Point_In_UCS_X', 'Right_Gaze_Point_In_UCS_Y', 'Right_Gaze_Point_In_UCS_Z',
                 'Left_Gaze_Point_Validity', 'Right_Gaze_Point_Validity', 'Left_Pupil_Diameter',
                 'Right_Pupil_Diameter', 'Left_Pupil_Validity', 'Right_Pupil_Validity',
                 'Left_Gaze_Origin_In_UCS_X', 'Left_Gaze_Origin_In_UCS_Y', 'Left_Gaze_Origin_In_UCS_Z',
                 'Right_Gaze_Origin_In_UCS_X', 'Right_Gaze_Origin_In_UCS_Y', 'Right_Gaze_Origin_In_UCS_Z',
                 'Left_Gaze_Origin_In_TBC_X', 'Left_Gaze_Origin_In_TBC_Y', 'Left_Gaze_Origin_In_TBC_Z',
                 'Right_Gaze_Origin_In_TBC_X', 'Right_Gaze_Origin_In_TBC_Y', 'Right_Gaze_Origin_In_TBC_Z',
                 'Left_Gaze_Origin_Validity', 'Right_Gaze_Origin_Validity', 'System_Time', 'Date_time',
                 'Gaze_Vector_Distance', 'Filepath'])
        f.close()

        label['text'] = "Θέλετε να παράγετε heatmaps;"
        buttonDisplay.place_forget()
        buttonExp.place_forget()
        buttonVid.place_forget()
        buttonGaze.place_forget()
        buttonDA.place_forget()

        appExec()


def images():
    label.config(text='Eye tracking κατά την παρακολούθηση εικόνων')
    buttonBack6.place_forget()
    buttonBack.place(relx=0, relwidth=0.2, relheight=1)
    appear_frame.config(bg='white')
    appear_frame.place(relx=0.5, rely=0.2, relwidth=0.7, relheight=0.6, anchor='n')

    label4.place(relx=0.1, relwidth=0.8, rely=0.1, relheight=0.6)

    entry.place(relx=0.1, relwidth=0.5, rely=0.7, relheight=0.2)
    buttonSave['command'] = get_name2
    buttonSave.place(relx=0.6, relwidth=0.3, rely=0.7, relheight=0.2)


def video_exp():
    label.config(text='Eye tracking κατά την παρακολούθηση βίντεο')
    buttonBack6.place_forget()
    buttonBack.place(relx=0, relwidth=0.2, relheight=1)
    appear_frame.config(bg='white')
    appear_frame.place(relx=0.5, rely=0.2, relwidth=0.7, relheight=0.6, anchor='n')

    label4.place(relx=0.1, relwidth=0.8, rely=0.1, relheight=0.6)

    entry.place(relx=0.1, relwidth=0.5, rely=0.7, relheight=0.2)
    buttonSave['command'] = get_name3
    buttonSave.place(relx=0.6, relwidth=0.3, rely=0.7, relheight=0.2)


def draw_image():
    global index_vi
    global csv_fil
    root.image_list = []

    if var1.get() == 0 and var2.get() == 0 and var3.get() == 0:
        image()
    else:
        buttonBack5.place_forget()
        home_directory = os.path.expanduser("~")

        root.filename_im = filedialog.askopenfilename(initialdir=home_directory, title='Αρχείο Εικόνας (png)',
                                                  filetypes=(("png Image", "*.png"),))


        root.filename_csv_fil = filedialog.askopenfilename(initialdir=home_directory,
                                                       title='Αρχείο Csv (Filtered Data)',
                                                       filetypes=(("csv File", "Filtered_Eye_Data*.*"),))

        root.filename_csv_fix = filedialog.askopenfilename(initialdir=home_directory, title='Αρχείο Csv (Fixations)',
                                                           filetypes=(("csv File", "Fixation_Data*.*"),))


        if len(root.filename_csv_fil) == 0 or len(root.filename_im) == 0:
            image()

        else:
            if var1.get() == 1:
                root.image_list.append(1)
            elif var1.get() == 0:
                root.image_list.append(0)
            if var2.get() == 1:
                root.image_list.append(1)
            elif var2.get() == 0:
                root.image_list.append(0)
            if var3.get() == 1:
                root.image_list.append(1)
            elif var3.get() == 0:
                root.image_list.append(0)

            buttonImGet.place_forget()
            c1.place_forget()
            c3.place_forget()
            df = pd.read_csv(root.filename_csv_fix, encoding='ISO-8859-7')
            if 'Fixation Start' in df.columns:
                label.config(text='Θέλετε να επιλέξετε ώρα έναρξης/λήξης;')
                button3.place(relx=0.4, rely=0, relwidth=0.2, relheight=0.4)
                button2.place(relx=0.4, rely=0.5, relwidth=0.2, relheight=0.4)
            else:
                no_time()


def grab_date():
    global index_vi
    global date_start
    global im
    global date_finish
    global index_start
    global index_stop
    global options
    date_start = cal.get()
    date_finish = cal2.get()
    index_start = cal.current()
    index_stop = cal2.current()


    if index_start >= index_stop:
        label.config(text="Μη αποδεκτές ώρες έναρξης/λήξης")
    else:
        cal.place_forget()
        cal2.place_forget()
        buttonDate.place_forget()
        label5.place_forget()
        label6.place_forget()
        buttonDate.place_forget()
        if index_vi == 0:
            draw_vid()
        elif index_vi == 1:
            if root.image_list[0] == 1:
                df = pd.read_csv(root.filename_csv_fix, encoding='ISO-8859-7')
                df1 = pd.read_csv(root.filename_csv_fil, encoding='ISO-8859-7')
                im = root.filename_im
                x = []
                y = []
                t = []
                if date_finish == None and date_start == None:
                    x = df['X Coordinate'].tolist()
                    y = df['Y Coordinate'].tolist()
                    t = df['Fixation Duration'].tolist()
                else:
                    x_l = df['X Coordinate'].tolist()
                    y_l = df['Y Coordinate'].tolist()
                    t_l = df['Fixation Duration'].tolist()
                    date_time = df1['Date_time'].tolist()
                    fix_start = df['Fixation Start'].tolist()
                    fix_end = df['Fixation End'].tolist()

                    for i in range(len(fix_start)):
                        if fix_start[i] > date_time[index_start] and fix_end[i] < date_time[index_stop]:
                            x.append(x_l[i])
                            y.append(y_l[i])
                            t.append(t_l[i])
                FixationScanIm(x, y, t, root.filename_im)
            if root.image_list[1] == 1:
                df = pd.read_csv(root.filename_csv_fix, encoding='ISO-8859-7')
                df1 = pd.read_csv(root.filename_csv_fil, encoding='ISO-8859-7')
                im = root.filename_im
                x = []
                y = []
                if date_finish == None and date_start == None:
                    x = df['X Coordinate'].tolist()
                    y = df['Y Coordinate'].tolist()
                else:
                    x_l = df['X Coordinate'].tolist()
                    y_l = df['Y Coordinate'].tolist()
                    date_time = df1['Date_time'].tolist()
                    fix_start = df['Fixation Start'].tolist()
                    fix_end = df['Fixation End'].tolist()
                    for i in range(len(fix_start)):
                        if fix_start[i] > date_time[index_start] and fix_end[i] < date_time[index_stop]:
                            x.append(x_l[i])
                            y.append(y_l[i])


            if root.image_list[2] == 1:
                threading.Thread(target=get_map_Im).start()

            else:
                label.config(text='Αρχικό Μενού')
                buttonEye.place(relx=0.4, rely=0, relwidth=0.2, relheight=0.4)
                buttonData.place(relx=0.4, rely=0.5, relwidth=0.2, relheight=0.4)

    options = []


def timer():
    buttonBack5.place_forget()
    button1.place_forget()
    button2.place_forget()

    label.config(text="Εισάγετε ώρες έναρξης και λήξης")
    df = pd.read_csv(root.filename_csv, encoding='ISO-8859-7')
    time_list =df['Date_time'].tolist()
    for i in range(len(time_list)):
        options.append(datetime.fromtimestamp(time_list[i]))
    cal.set(options[0])
    cal2.set(options[0])
    cal['value'] = options
    cal['state'] = 'readonly'
    cal2['value'] = options
    cal2['state'] = 'readonly'
    cal.place(relx=0.1, relwidth=0.3, rely=0.4, relheight=0.2)
    cal2.place(relx=0.6, relwidth=0.3, rely=0.4, relheight=0.2)
    label5.place(relx=0.1, relwidth=0.3, rely=0.1, relheight=0.2)
    label6.place(relx=0.6, relwidth=0.3, rely=0.1, relheight=0.2)
    buttonDate.place(relx=0.8, relwidth=0.2, relheight=1)


def timer_im():
    button3.place_forget()
    button2.place_forget()

    label.config(text="Εισάγετε ώρες έναρξης και λήξης")

    df = pd.read_csv(root.filename_csv_fil, encoding='ISO-8859-7')
    time_list =df['Date_time'].tolist()
    for i in range(len(time_list)):
        options.append(datetime.fromtimestamp(time_list[i]))
    cal.set(options[0])
    cal2.set(options[0])
    cal['value'] = options
    cal['state'] = 'readonly'
    cal2['value'] = options
    cal2['state'] = 'readonly'
    cal.place(relx=0.1, relwidth=0.3, rely=0.4, relheight=0.2)
    cal2.place(relx=0.6, relwidth=0.3, rely=0.4, relheight=0.2)
    label5.place(relx=0.1, relwidth=0.3, rely=0.1, relheight=0.2)
    label6.place(relx=0.6, relwidth=0.3, rely=0.1, relheight=0.2)
    buttonDate.place(relx=0.8, relwidth=0.2, relheight=1)


def no_time():
    global date_start
    global date_finish
    date_start = None
    date_finish = None
    button3.place_forget()
    button2.place_forget()

    if index_vi == 0:
        draw_vid()
    elif index_vi == 1:
        df = pd.read_csv(root.filename_csv_fix, encoding='ISO-8859-7')
        df1 = pd.read_csv(root.filename_csv_fil, encoding='ISO-8859-7')
        im = root.filename_im
        x_l = df['X Coordinate'].tolist()
        y_l = df['Y Coordinate'].tolist()
        t_l = df['Fixation Duration'].tolist()
        x = []
        y = []
        t = []

        if 'Fixation Start' not in df.columns:
            file_fix = df['Filename'].to_list()
            for i in range(len(file_fix)):
                if Path(root.filename_im).stem == Path(file_fix[i]).stem:
                    x.append(x_l[i])
                    y.append(y_l[i])
                    t.append(t_l[i])
        else:
            for i in range(len(x_l)):
                x.append(x_l[i])
                y.append(y_l[i])
                t.append(t_l[i])

        if root.image_list[0] == 1:
            FixationScanIm(x, y, t, root.filename_im)
        if root.image_list[1] == 1:
            pass
        if root.image_list[2] == 1:
            threading.Thread(target=get_map_Im).start()
        else:
            label.config(text='Αρχικό Μενού')
            buttonEye.place(relx=0.4, rely=0, relwidth=0.2, relheight=0.4)
            buttonData.place(relx=0.4, rely=0.5, relwidth=0.2, relheight=0.4)


def start_experiment():
    label.config(text='Παρακολούθηση Οφθαλμού με χρήση του πληκτρολογίου')
    buttonBack6.place_forget()
    buttonBack.place(relx=0, relwidth=0.2, relheight=1)
    appear_frame.config(bg='white')
    appear_frame.place(relx=0.5, rely=0.2, relwidth=0.7, relheight=0.6, anchor='n')

    # Makes Project Folder
    label4.place(relx=0.1, relwidth=0.8, rely=0.1, relheight=0.6)
    entry.place(relx=0.1, relwidth=0.5, rely=0.7, relheight=0.2)

    buttonSave['command'] = get_name
    buttonSave.place(relx=0.6, relwidth=0.3, rely=0.7, relheight=0.2)


def video():
    global index_vi
    index_vi = 0
    home_directory = os.path.expanduser("~")

    root.filename_vid = filedialog.askopenfilename(initialdir=home_directory, title='Select Video File',
                                                   filetypes=(("mp4 Video", "*.mp4"), ("flv Video", "*.flv"),
                                                              ("mov Video", "*.mov"), ("wmv Video", "*.wmv"),
                                                              ("avi Video", "*.avi")))

    root.filename_csv = filedialog.askopenfilename(initialdir=home_directory, title='Select Csv File',
                                                   filetypes=(("csv File", "Filtered_Eye_Data*.*"),))

    if len(root.filename_csv) == 0 or len(root.filename_vid) == 0:
        back2()
    else:
        df = pd.read_csv(root.filename_csv, encoding='ISO-8859-7')
        files = df['Filepath']
        if isinstance(files[0], float) == True:
            buttonBack2.place_forget()
            buttonBack5.place(relx=0, relwidth=0.2, relheight=1)
            buttonVideo.place_forget()
            buttonImage.place_forget()

            label['text'] = 'Θέλετε να επιλέξετε ώρες έναρξης/λήξης;'
            button1.place(relx=0.4, rely=0, relwidth=0.2, relheight=0.4)
            button2.place(relx=0.4, rely=0.5, relwidth=0.2, relheight=0.4)
            button1['command'] = timer
        elif isinstance(files[0], str) == True:
            buttonBack2.place_forget()
            buttonBack5.place(relx=0, relwidth=0.2, relheight=1)
            buttonVideo.place_forget()
            buttonImage.place_forget()
            no_time()


def image():
    global index_vi
    index_vi = 1
    buttonVideo.place_forget()
    buttonImage.place_forget()
    label.config(text="Επιλέξτε")

    c1.config(font=('Helvetica bold', 12))
    c3.config(font=('Helvetica bold', 12))
    c1.place(relx=0.2, relwidth=0.6, rely=0, relheight=0.5)
    c3.place(relx=0.2, relwidth=0.6, rely=0.5, relheight=0.5)
    buttonImGet.place(relx=0.8, relwidth=0.2, relheight=1)
    button1.place_forget()
    buttonBack2.place_forget()
    buttonBack5.place(relx=0, relwidth=0.2, relheight=1)


def back2():
    buttonBack2.place_forget()
    label.config(text="Αρχικό Μενού")
    buttonVideo.place_forget()
    buttonImage.place_forget()
    buttonEye.place(relx=0.4, rely=0, relwidth=0.2, relheight=0.4)
    buttonData.place(relx=0.4, rely=0.5, relwidth=0.2, relheight=0.4)


def back5():
    buttonBack5.place_forget()
    buttonBack2.place(relx=0, relwidth=0.2, relheight=1)
    label.config(text='Εισάγετε Εικόνα/Βίντεο:')
    c1.place_forget()
    c3.place_forget()
    buttonImGet.place_forget()
    buttonVideo.place(relx=0.3, rely=0, relwidth=0.4, relheight=0.4)
    buttonImage.place(relx=0.3, rely=0.5, relwidth=0.4, relheight=0.4)


def back3():
    buttonBack3.place_forget()
    label.config(text="Αρχικό Μενού")
    buttonYes.place_forget()
    buttonNo.place_forget()
    buttonEye.place(relx=0.4, rely=0, relwidth=0.2, relheight=0.4)
    buttonData.place(relx=0.4, rely=0.5, relwidth=0.2, relheight=0.4)


def back6():
    mid_frame.place(relx=0.5, rely=0.2, relwidth=0.75, relheight=0.25, anchor='n')
    outly1.place(relx=0.1, rely=0.1, relwidth=0.1, relheight=0.4, anchor='n')
    outly2.place(relx=0.85, rely=0.1, relwidth=0.1, relheight=0.4)
    outly3.place(relx=0.05, rely=0.45, relwidth=0.9, relheight=0.1)

    buttonBack6.place_forget()
    buttonBack3.place(relx=0, relwidth=0.2, relheight=1)
    buttonDisplay.place_forget()
    buttonExp.place_forget()
    buttonGaze.place_forget()
    buttonDA.place_forget()
    buttonVid.place_forget()
    label.config(text="Είναι η πρώτη φορά που χρησιμοποιείτε τον eye tracker;")
    buttonYes.place(relx=0.4, rely=0, relwidth=0.2, relheight=0.4)
    buttonNo.place(relx=0.4, rely=0.5, relwidth=0.2, relheight=0.4)


def data():
    buttonEye.place_forget()
    buttonData.place_forget()
    buttonVideo.place(relx=0.3, rely=0, relwidth=0.4, relheight=0.4)
    buttonImage.place(relx=0.3, rely=0.5, relwidth=0.4, relheight=0.4)
    label.config(text='Εισάγετε εικόνα/βίντεο')
    buttonBack2.place(relx=0, relwidth=0.2, relheight=1)


def Eye():
    global eyetracker
    global eyetrackers
    eyetrackers = tr.find_all_eyetrackers()  # Finds all eye tracker objects
    if len(eyetrackers) == 0:
        messagebox.showinfo("Σφάλμα", "Δεν εντοπίστηκα συσκευή eye tracking. Παρακαλώ συνδέστε στην θύρα usb και επαναλάβετε")
        sys.exit(0)
    eyetracker = eyetrackers[0]
    label.config(text='Είναι η πρώτη φορά που χρησιμοποιείται τον eye tracker;')
    buttonEye.place_forget()
    buttonData.place_forget()
    buttonYes.place(relx=0.4, rely=0, relwidth=0.2, relheight=0.4)
    buttonNo.place(relx=0.4, rely=0.5, relwidth=0.2, relheight=0.4)
    buttonBack3.place(relx=0, relwidth=0.2, relheight=1)


def yes():
    call_eyetracker_manager(eyetracker.address, "displayarea")
    second_q()


def cal():
    call_eyetracker_manager(eyetracker.address, "usercalibration")


root = tk.Tk()
root.resizable(False, False)

canvas = tk.Canvas(root, height=600, width=800)
canvas.pack()
col = '#D1EEEE'

background_image = tk.PhotoImage(file='Program2.png')
background_label = tk.Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)

outly6 = tk.Frame(root, bg='black', bd=5)
frame2 = tk.Frame(root, bg='black', bd=5)
frame = tk.Frame(root, bg=col, bd=5)
frame.place(relx=0.5, rely=0.1, relwidth=0.75, relheight=0.1, anchor='n')


mid_frame = tk.Frame(root, bg='white', bd=5)
mid_frame.place(relx=0.5, rely=0.2, relwidth=0.75, relheight=0.25, anchor='n')

buttonEye = tk.Button(mid_frame, text='Eye tracking', command=Eye)
buttonData = tk.Button(mid_frame, text='Δεδομένα', command=data)
buttonEye.place(relx=0.4, rely=0, relwidth=0.2, relheight=0.4)
buttonData.place(relx=0.4, rely=0.5, relwidth=0.2, relheight=0.4)

button1 = tk.Button(mid_frame, text='Ναι', command=timer)
button2 = tk.Button(mid_frame, text='Όχι', command=no_time)
button3 = tk.Button(mid_frame, text='Ναι', command=timer_im)

buttonYes = tk.Button(mid_frame, text='Ναι', command=yes)
buttonNo = tk.Button(mid_frame, text="Όχι", command=second_q)

outly1 = tk.Frame(root, bg=col, bd=5)
outly1.place(relx=0.1, rely=0.1, relwidth=0.1, relheight=0.4, anchor='n')
outly2 = tk.Frame(root, bg=col, bd=5)
outly2.place(relx=0.85, rely=0.1, relwidth=0.1, relheight=0.4)
outly3 = tk.Frame(root, bg=col, bd=5)
outly3.place(relx=0.05, rely=0.45, relwidth=0.9, relheight=0.1)
outly4 = tk.Frame(root, bg=col, bd=5)
outly5 = tk.Frame(root, bg=col, bd=5)


buttonStartTime = tk.Button(mid_frame, text="Έναρξη")
buttonEndTime = tk.Button(mid_frame, text="Λήξη")
buttonBack2 = tk.Button(outly3, text="Επιστροφή", command=back2)
buttonBack3 = tk.Button(outly3, text="Επιστροφή", command=back3)
buttonBack6 = tk.Button(outly3, text="Επιστροφή", command=back6)
buttonBack5 = tk.Button(outly3, text="Επιστροφή", command=back5)

entry2 = tk.Entry(mid_frame)

buttonY = tk.Button(mid_frame, text="Ναι", command=start_progress)
buttonN = tk.Button(mid_frame, text="Όχι", command=end)

buttonY2 = tk.Button(mid_frame, text="Ναι", command=draw2)
buttonN2 = tk.Button(mid_frame, text="Όχι", command=end)

buttonDate = tk.Button(outly3, text='Ok', command=grab_date)

options = []
var4 = StringVar(mid_frame)
var5 = StringVar(mid_frame)
cal = Combobox(mid_frame, textvariable=var4, values=options)
cal2 = Combobox(mid_frame, textvariable=var5, values=options)

appear_frame = tk.Frame(root, bg='white', bd=5)
frame_back = tk.Frame(appear_frame, bg='white', bd=5)

label = tk.Label(frame, text="Καλώς ήρθατε", font=10, bg=col)
label.place(relheight=1, relwidth=1)

label2 = tk.Label(frame, text="????????", font=40, bg=col)
label3 = tk.Label(appear_frame, bg="white")
label4 = tk.Label(appear_frame, text="Εισάγετε το ονοματεπώνυμο του συμμετέχοντα:")
label5 = tk.Label(mid_frame, bg="white", text='Ώρα Έναρξης')
label6 = tk.Label(mid_frame, bg='white', text='Ώρα Λήξης')
entry = tk.Entry(appear_frame)

buttonDisplay = tk.Button(mid_frame, text='1.Εμφάνιση Ιδιοτήτων Eye tracker', command=dip_properties)
buttonDA = tk.Button(mid_frame, text="2.Eye Tracker Manager", command=yes)
buttonGaze = tk.Button(mid_frame, text="3. Ελεύθερο Πείραμα", command=start_experiment)
buttonExp = tk.Button(mid_frame, text="4. Πείραμα Εικόνων", command=images)
buttonVid = tk.Button(mid_frame, text="5. Πείραμα Βίντεο", command=video_exp)
buttonBack = tk.Button(outly3, text="Επιστροφή", command=back)
buttonSave = tk.Button(appear_frame, text="Αποθήκευση", font=40)

progress = Progressbar(mid_frame, orient=HORIZONTAL,
                       length=100, mode='determinate')

buttonVideo = tk.Button(mid_frame, text='Επιλέξτε Βίντεο', command=video)
buttonImage = tk.Button(mid_frame, text='Επιλέξτε εικόνα', command=image)

var1 = IntVar()
var2 = IntVar()
var3 = IntVar()

c1 = Checkbutton(mid_frame, text="Fixation-Scanpath", variable=var1)
c3 = Checkbutton(mid_frame, text="Heatmap", variable=var3)

buttonImGet = tk.Button(outly3, text="Οκ", command=draw_image)


if __name__ == "__main__":
    home_directory = os.path.expanduser("~")  # Finds home directory (cross platform)
    global index_gaze
    index_gaze = 0
    global filtered_eye_data
    global opened
    global seconds
    global index_start
    global index_stop
    opened = 0
    global raw_data_name
    global date_start
    global date_finish
    global im
    global csv_fil
    global end_map
    global project_directory
    global filepath
    global eyetrackers
    global eyetracker
    global image_dir
    global fixation_dir
    global heatmap_dir
    global scanpath_dir
    global index_vi
    filepath = ''

    root.mainloop()


