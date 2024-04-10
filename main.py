import cv2
import mediapipe as mp
import os
import random
import pygame

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize Pygame
pygame.init()

# Initialize OpenCV VideoCapture
cap = cv2.VideoCapture(0)

# Previous landmark positions
prev_landmarks = None

# Folder path containing songs
songs_folder = "/home/user/Downloads/Agnyaathavaasi-320kbps-MassTamilan"
songs = os.listdir(songs_folder)
current_song_index = 0

# Load the first song
pygame.mixer.music.load(os.path.join(songs_folder, songs[current_song_index]))

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect hand landmarks
    results = hands.process(rgb_frame)
    
    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        # Get landmarks for the first hand (assuming only one hand is in frame)
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Extract landmark positions
        landmark_positions = []
        for landmark in hand_landmarks.landmark:
            landmark_positions.append((landmark.x, landmark.y, landmark.z))
        
        # Previous landmarks available
        if prev_landmarks:
            # Calculate differences between current and previous landmark positions
            differences = [(cur[0] - prev[0], cur[1] - prev[1], cur[2] - prev[2]) for cur, prev in zip(landmark_positions, prev_landmarks)]
            
            # Recognize gestures based on differences
            thumb, index, middle, ring, pinky = differences[4], differences[8], differences[12], differences[16], differences[20]
            
            # Check for play/pause gesture (thumb and index finger close)
            if thumb[0] < -0.02 and index[0] < -0.02:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.pause()
                else:
                    pygame.mixer.music.unpause()
            # Check for next song gesture (index and middle fingers extended)
            elif index[1] > 0.02 and middle[1] > 0.02:
                current_song_index = (current_song_index + 1) % len(songs)
                pygame.mixer.music.load(os.path.join(songs_folder, songs[current_song_index]))
                pygame.mixer.music.play()
            # Check for previous song gesture (ring and pinky fingers extended)
            elif ring[1] > 0.02 and pinky[1] > 0.02:
                current_song_index = (current_song_index - 1) % len(songs)
                pygame.mixer.music.load(os.path.join(songs_folder, songs[current_song_index]))
                pygame.mixer.music.play()
        
        # Update previous landmarks
        prev_landmarks = landmark_positions
        
    # Display the frame
    cv2.imshow('Hand Gesture Music Control', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
