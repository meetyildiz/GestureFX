import cv2
import mediapipe as mp
from src.detection import detectHandsLandmarks, countFingers, recognizeGestures, get_action_index, action2channel
from src.graphics import annotate, drawBasicGrid, draw_cursor, draw_keys, draw_triger_push, draw_sensitive_push, draw_sticky_push, draw_mixing_push, draw_effect_push
from src.midi import send_triger_push_midi, send_sensitive_push_midi, send_fader_midi, send_effect_midi
from datetime import datetime
# Initialize the mediapipe hands class.
mp_hands = mp.solutions.hands
 
# Set up the Hands functions for images and videos.
#hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
hands_videos = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.9)
 
# Initialize the mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils


def run_cam():
    from src.settings import (
        screen_modes_list,
        screen_modes_status,
        last_action_index,
        last_channel_index,
        last_hands_gestures,
        last_channel_index,
        last_screen_mode,
        current_screen_mode,
        drumrack_grid_size,
        pianoroll_grid_size,
        session_grid_size,
        mixing_grid_size,
        effects_grid_size,
        effects_cache
    )
    
    # Initialize the VideoCapture object to read from the webcam.
    camera_video = cv2.VideoCapture(0)
    camera_video.set(3,2500)
    camera_video.set(4,2500)

    # Create named window for resizing purposes.
    #cv2.namedWindow('Counted Fingers Visualization', cv2.WINDOW_NORMAL)


    # Iterate until the webcam is accessed successfully.
    while camera_video.isOpened():
        print("/" * 100)
        print("LOG DATE: " + str(datetime.now()))
        # Read a frame.
        ok, frame = camera_video.read()
        
        # Check if frame is not read properly then continue to the next iteration to read the next frame.
        if not ok:
            continue
        
        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)

        cv2.putText(frame, current_screen_mode , (200, (1) * 60),
                        cv2.FONT_HERSHEY_PLAIN, 4, (0,255,0), 5)
        cv2.putText(frame, str(screen_modes_status[current_screen_mode]["PAGE"]) , (200, (2) * 60),
                        cv2.FONT_HERSHEY_PLAIN, 4, (0,255,0), 5)


        frame, results = detectHandsLandmarks(frame, hands_videos, display=False)

        # Check if the hands landmarks in the frame are detected.
        if results.multi_hand_landmarks:

            # Count the number of fingers up of each hand in the frame.
            frame, fingers_statuses, count = countFingers(frame, results, display=False, draw=False)
        else:
            fingers_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False,
                        'RIGHT_PINKY': False, 'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False,
                        'LEFT_RING': False, 'LEFT_PINKY': False}
            count = {'RIGHT': 0, 'LEFT': 0}
        
        # Visualize the counted fingers.
        #frame = annotate(frame, results, fingers_statuses, count, display=False)

        frame, hands_gestures = recognizeGestures(frame, fingers_statuses, count, draw=False, display=False)
        frame, cursor = draw_cursor(frame, results)
        #gesture göre değiştir

        if hands_gestures["RIGHT"] == "CONTROL 1 SIGN" and last_hands_gestures["RIGHT"] != "CONTROL 1 SIGN" and screen_modes_list.index(current_screen_mode) != len(screen_modes_list)-1:
            current_screen_mode = screen_modes_list[screen_modes_list.index(current_screen_mode)+1]

        if hands_gestures["LEFT"] == "CONTROL 1 SIGN" and last_hands_gestures["LEFT"] != "CONTROL 1 SIGN"  and screen_modes_list.index(current_screen_mode) != 0:
            current_screen_mode = screen_modes_list[screen_modes_list.index(current_screen_mode)-1]

        if hands_gestures["RIGHT"] == "CONTROL 2 SIGN" and last_hands_gestures["RIGHT"] != "CONTROL 2 SIGN":
            screen_modes_status[current_screen_mode]["PAGE"] += 1

        if hands_gestures["LEFT"] == "CONTROL 2 SIGN" and last_hands_gestures["LEFT"] != "CONTROL 2 SIGN" and screen_modes_status[current_screen_mode]["PAGE"] != 1:
            screen_modes_status[current_screen_mode]["PAGE"] -= 1





        if current_screen_mode == "DRUMRACK":
            
            color = (24, 120, 174)

            frame, grid = drawBasicGrid(frame, grid_size=drumrack_grid_size, color=color)
            frame, notes = draw_keys(frame, grid, page=screen_modes_status[current_screen_mode]["PAGE"])
            
            action_index = get_action_index(grid, cursor, hands_gestures)
            frame = draw_triger_push(frame, grid, cursor, hands_gestures, color)

            channel_index = action2channel(grid_size=drumrack_grid_size, action_index=action_index, notes=notes)
            send_triger_push_midi(last_channel_index, channel_index)
            

        elif current_screen_mode == "PIANOROLL":
            
            color = (47, 78, 100)

            frame, grid = drawBasicGrid(frame, grid_size=pianoroll_grid_size, color = color)
            frame, notes = draw_keys(frame, grid, page=screen_modes_status[current_screen_mode]["PAGE"])
            
            action_index = get_action_index(grid, cursor, hands_gestures)
            frame = draw_sensitive_push(frame, grid, cursor, hands_gestures, action_index, color)

            channel_index = action2channel(grid_size=pianoroll_grid_size, action_index=action_index, notes=notes)
            send_sensitive_push_midi(last_channel_index, channel_index, action_index)
            

        elif current_screen_mode == "SESSION":
            
            color = (36, 12, 200)

            frame, grid = drawBasicGrid(frame, grid_size=session_grid_size, color=color)
            frame, notes = draw_keys(frame, grid, page=screen_modes_status[current_screen_mode]["PAGE"], draw_notes=False)
            
            action_index = get_action_index(grid, cursor, hands_gestures)
            frame = draw_sticky_push(frame, grid, cursor, hands_gestures, color)

            channel_index = action2channel(grid_size=session_grid_size, action_index=action_index, notes=notes)
            send_triger_push_midi(last_channel_index, channel_index)
            
        elif current_screen_mode == "MIXING":
            
            color = (145, 100, 94)

            frame, grid = drawBasicGrid(frame, grid_size=mixing_grid_size, color=color)
            frame, notes = draw_keys(frame, grid, page=screen_modes_status[current_screen_mode]["PAGE"], draw_notes=False)
            
            action_index = get_action_index(grid, cursor, hands_gestures)
            frame = draw_mixing_push(frame, grid, cursor, hands_gestures, action_index, color)

            channel_index = action2channel(grid_size=mixing_grid_size, action_index=action_index, notes=notes)
            send_fader_midi(last_channel_index, channel_index, action_index)

        elif current_screen_mode == "EFFECTS":
            
            color = (120, 10, 94)

            #frame, _ = draw_effect_cursor(frame, results)

            frame, grid = drawBasicGrid(frame, grid_size=effects_grid_size, color=color)
            frame, notes = draw_keys(frame, grid, page=screen_modes_status[current_screen_mode]["PAGE"], draw_notes=False)
            
            action_index = get_action_index(grid, cursor, hands_gestures)
            frame = draw_triger_push(frame, grid, cursor, hands_gestures, color)

            channel_index = action2channel(grid_size=effects_grid_size, action_index=action_index, notes=notes)
            print("A"*100)
            print(notes)
            print(channel_index)
            print(action_index)
            print(grid)
            print('+'*100)
            #print(grid_note_lookup)
            #img, grid, cursors, gestures, color

            frame = draw_effect_push(frame, grid, cursor, hands_gestures, color)


            send_effect_midi(channel_index, action_index)
            
            
        else:
            pass

        last_hands_gestures = hands_gestures
        last_action_index = action_index
        last_channel_index = channel_index
        last_screen_mode = current_screen_mode
        last_cursor = cursor


        # UI up
        cv2.imshow('GestureFX', frame)
        cv2.setWindowProperty('GestureFX', cv2.WND_PROP_TOPMOST, 1)

        # Wait for 1ms. If a key is pressed, re treive the ASCII code of the key.
        k = cv2.waitKey(1) & 0xFF
        
        # Check if 'ESC' is pressed and break the loop.
        if(k == 27):
            break
        
        print("")
        print("")
    # Release the VideoCapture Object and close the windows.
    camera_video.release()
    cv2.destroyAllWindows()




