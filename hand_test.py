#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import os
import copy
import argparse
import pygame
import cv2 as cv
import numpy as np
import mediapipe as mp
from collections import Counter, deque
import itertools

from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", help='min_detection_confidence', type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", help='min_tracking_confidence', type=int, default=0.5)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    use_brect = True

    # Camera preparation
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # Read labels
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

    with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [row[0] for row in point_history_classifier_labels]

    # FPS Measurement
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history
    finger_gesture_history = deque(maxlen=history_length)

    # Mode for controlling volume
    mode = 0

    # Music setup
    pygame.init()
    pygame.mixer.init()

    songs_folder = "Songs"
    songs = os.listdir(songs_folder)
    current_song_index = 0

    pygame.mixer.music.load(os.path.join(songs_folder, songs[current_song_index]))
    pygame.mixer.music.play()

    volume = 0.5

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end)
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
                logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 0:
                    pygame.mixer.music.pause()
                elif hand_sign_id == 1:
                    pygame.mixer.music.unpause()
                elif hand_sign_id == 2:
                    volume_up()
                elif hand_sign_id == 3:
                    volume_down()

                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)
    return x, y, w, h


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_list = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_list.append([landmark_x, landmark_y])

    return landmark_list


def pre_process_landmark(landmark_list):
    x_list, y_list = zip(*landmark_list)
    pre_processed_landmark_list = list(itertools.chain.from_iterable([x_list, y_list]))
    return pre_processed_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    flatten_point_history = list(itertools.chain.from_iterable(point_history))
    flatten_point_history = list(
        map(
            lambda landmark: [
                min(int(landmark[0] * image_width), image_width - 1),
                min(int(landmark[1] * image_height), image_height - 1),
            ],
            flatten_point_history,
        )
    )
    return flatten_point_history


def logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list):
    with open('log.csv', mode='a') as f:
        writer = csv.writer(f)
        if mode == 0:
            writer.writerow([number] + pre_processed_landmark_list)
        elif mode == 1:
            writer.writerow([number] + pre_processed_point_history_list)
        elif mode == 2:
            writer.writerow([number])


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        x, y, w, h = brect
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
    return image


def draw_landmarks(image, landmark_list):
    for landmark in landmark_list:
        cv.circle(image, (int(landmark[0]), int(landmark[1])), 5, (0, 255, 0), thickness=-1, lineType=cv.FILLED)
    return image


def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    info_text_color = (0, 255, 0)
    info_text_pos = (brect[0], brect[1] - 10)
    info_text_scale = 0.6
    info_text_thickness = 1
    info_text_font = cv.FONT_HERSHEY_SIMPLEX
    info_text_margin = 10

    cv.putText(image, f'{hand_sign_text}', (info_text_pos[0], info_text_pos[1] - info_text_margin * 2), info_text_font,
               info_text_scale, info_text_color, info_text_thickness, lineType=cv.LINE_AA)
    cv.putText(image, f'{finger_gesture_text}', (info_text_pos[0], info_text_pos[1] - info_text_margin), info_text_font,
               info_text_scale, info_text_color, info_text_thickness, lineType=cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for i, landmark_point in enumerate(point_history):
        cv.circle(image, (int(landmark_point[0]), int(landmark_point[1])), 1, (255, 255, 0), thickness=-1,
                  lineType=cv.FILLED)
    return image


def draw_info(image, fps, mode, number):
    info_text_color = (0, 255, 0)
    info_text_pos = (10, 30)
    info_text_scale = 0.6
    info_text_thickness = 1
    info_text_font = cv.FONT_HERSHEY_SIMPLEX

    cv.putText(image, f'FPS: {fps:.2f}', (info_text_pos[0], info_text_pos[1] - 30), info_text_font, info_text_scale,
               info_text_color, info_text_thickness, lineType=cv.LINE_AA)

    mode_text = ''
    if mode == 0:
        mode_text = 'KEYPOINT'
    elif mode == 1:
        mode_text = 'POINT_HISTORY'
    elif mode == 2:
        mode_text = 'NORMAL'
    cv.putText(image, f'MODE: {mode_text}', (info_text_pos[0], info_text_pos[1] - 60), info_text_font, info_text_scale,
               info_text_color, info_text_thickness, lineType=cv.LINE_AA)

    number_text = ''
    if 0 <= number <= 9:
        number_text = str(number)
    cv.putText(image, f'NUMBER: {number_text}', (info_text_pos[0], info_text_pos[1] - 90), info_text_font,
               info_text_scale, info_text_color, info_text_thickness, lineType=cv.LINE_AA)

    return image


def volume_up():
    global volume
    volume += 0.1
    if volume > 1.0:
        volume = 1.0
    pygame.mixer.music.set_volume(volume)


def volume_down():
    global volume
    volume -= 0.1
    if volume < 0.0:
        volume = 0.0
    pygame.mixer.music.set_volume(volume)


if __name__ == "__main__":
    main()




import time

# Add a cooldown period for gesture recognition
gesture_cooldown = 2  # Adjust this value as needed (in seconds)
last_gesture_time = time.time()

while True:
    fps = cvFpsCalc.get()

    # Process Key (ESC: end) #################################################
    key = cv.waitKey(10)
    if key == 27:  # ESC
        break
    number, mode = select_mode(key, mode)

    # Camera capture #####################################################
    ret, image = cap.read()
    if not ret:
        break
    image = cv.flip(image, 1)  # Mirror display
    debug_image = copy.deepcopy(image)

    # Detection implementation #############################################################
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    #  ####################################################################
    if results.multi_hand_landmarks is not None:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                              results.multi_handedness):
            # Bounding box calculation
            brect = calc_bounding_rect(debug_image, hand_landmarks)
            # Landmark calculation
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)

            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
            # Write to the dataset file
            logging_csv(number, mode, pre_processed_landmark_list,pre_processed_point_history_list)

            # Hand sign classification
            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            current_time = time.time()
            if current_time - last_gesture_time >= gesture_cooldown:
                # Process the gesture only if the cooldown period has elapsed
                if hand_sign_id == 0:
                    if pygame.mixer.music.get_busy():
                        pygame.mixer.music.pause()
                        prev_hand_sing_id = 0
                        print("PAUSED")
                elif hand_sign_id == 1:
                    if prev_hand_sing_id != 1:
                        pygame.mixer.music.unpause()
                        prev_hand_sing_id = 1
                        print("PLAY")
                elif hand_sign_id == 2:
                    if prev_hand_sing_id != 2: 
                        current_song_index +=1
                        if current_song_index >= len(songs):
                            current_song_index = 0
                        pygame.mixer.music.load(os.path.join(songs_folder, songs[current_song_index]))
                        pygame.mixer.music.play()
                        prev_hand_sing_id = 2
                        print(f"NEXT, {current_song_index}")
                elif hand_sign_id == 3:
                    if prev_hand_sing_id != 3:
                        current_song_index -=1
                        if current_song_index <= 0:
                            current_song_index = len(songs)-1
                        pygame.mixer.music.load(os.path.join(songs_folder, songs[current_song_index]))
                        pygame.mixer.music.play()
                        prev_hand_sing_id = 3
                        print(f"PREVIOUS, {current_song_index}")
                elif hand_sign_id == 4:
                    volume_up()
                elif hand_sign_id == 5:
                    volume_down()
                    
                # Update the last gesture time
                last_gesture_time = current_time
