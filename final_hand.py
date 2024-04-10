import csv
import os
import copy
import pygame
import itertools
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp
import tensorflow as tf
import math 

volume = 0.5

def keypoint_classifier(landmark_list, model_path='model/keypoint_classifier/keypoint_classifier.tflite', num_threads=1):
    # print("landmark", np.array(landmark_list).shape)
    interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_details_tensor_index = input_details[0]['index']
    interpreter.set_tensor(input_details_tensor_index, np.array([landmark_list], dtype=np.float32))
    interpreter.invoke()

    output_details_tensor_index = output_details[0]['index']
    result = interpreter.get_tensor(output_details_tensor_index)
    result_index = np.argmax(np.squeeze(result))

    return result_index

def main():
    use_brect = True
    
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
    
    # Read labels
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

    # Coordinate history
    history_length = 16
    keypoint_history = deque(maxlen=history_length)

    # Music setup
    pygame.init()
    pygame.mixer.init()

    songs_folder = "Songs"  
    songs = os.listdir(songs_folder)
    current_song_index = 0
    
    pygame.mixer.music.load(os.path.join(songs_folder, songs[current_song_index]))
    pygame.mixer.music.play()
    
    prev_hand_sing_id = 0

    while True:
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                if hand_sign_id == 0:
                    if pygame.mixer.music.get_busy():
                        pygame.mixer.music.pause()
                        prev_hand_sing_id = 0
                        print("PAUSE")

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
                    print("VOLUME_UP")
                    
                elif hand_sign_id == 5:
                    volume_down()
                    print("VOLUME_DOWN")

                finger_gesture_id = 0
                keypoint_history_len = len(pre_processed_landmark_list)
                if keypoint_history_len == (history_length * 2):
                    finger_gesture_id = keypoint_classifier(pre_processed_landmark_list)
                    

                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(debug_image, brect, handedness, keypoint_classifier_labels[hand_sign_id])
        else:
            keypoint_history.append([0, 0])

        cv.imshow('Hand Gesture Recognition', debug_image)
        # print(type(debug_image))

    cap.release()
    cv.destroyAllWindows()

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    # print("temp", np.array(temp_landmark_list).shape)

    return temp_landmark_list


def draw_landmarks(image, landmark_points):
    if len(landmark_points) > 0:
        thumb_base = landmark_points[8]
        index_base = landmark_points[4]

        # Calculate midpoint between thumb_base and index_base
        midpoint_x = (thumb_base[0] + index_base[0]) // 2
        midpoint_y = (thumb_base[1] + index_base[1]) // 2
        midpoint = (midpoint_x, midpoint_y)

        # Draw circle at midpoint
        # cv.circle(image, midpoint, 5, (255, 255, 255), -1)
        cv.circle(image, midpoint, 10, (255, 0, 0), cv.FILLED)
        cv.circle(image, thumb_base,10, (255, 0, 0), cv.FILLED)
        cv.circle(image, index_base, 10, (255, 0, 0), cv.FILLED)
        length = math.hypot(thumb_base[1]-thumb_base[0],index_base[1]-index_base[0])
        # print(length)
        volbar = np.interp(length, [50,150], [400,150])
        volper = np.interp(length, [50,150], [0,100] )

        if length < 60:
            cv.circle(image, midpoint, 10, (0, 0, 255), cv.FILLED)

        # Draw lines from thumb_base and index_base to midpoint
        cv.line(image, tuple(thumb_base), midpoint, (0, 255, 0), 3)
        cv.line(image, tuple(index_base), midpoint, (0, 255, 0), 3)
        cv.rectangle(image, (50, 150), (85, 400), (255, 0, 0), 3)
        cv.rectangle(image, (50,int(volbar)), (85, 400), (255, 0, 0), cv.FILLED)
        cv.putText(image, f'{int(volper)} %', (40,450), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)


    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image

def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ' : ' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv.LINE_AA)
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

if __name__ == '__main__':
    main()
