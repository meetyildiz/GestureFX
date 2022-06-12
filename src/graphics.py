import cv2
import time
import pygame
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from bisect import bisect
from src.notes import note_lookup




def annotate(image, results, fingers_statuses, count, display=True):
    '''
    This function will draw an appealing visualization of each fingers up of the both hands in the image.
    Args:
        image:            The image of the hands on which the counted fingers are required to be visualized.
        results:          The output of the hands landmarks detection performed on the image of the hands.
        fingers_statuses: A dictionary containing the status (i.e., open or close) of each finger of both hands. 
        count:            A dictionary containing the count of the fingers that are up, of both hands.
        display:          A boolean value that is if set to true the function displays the resultant image and 
                          returns nothing.
    Returns:
        output_image: A copy of the input image with the visualization of counted fingers.
    '''
    # Get the height and width of the input image.
    height, width, _ = image.shape
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Select the images of the hands prints that are required to be overlayed.
    ########################################################################################################################
    
    # Initialize a dictionaty to store the images paths of the both hands.
    # Initially it contains red hands images paths. The red image represents that the hand is not present in the image. 
    HANDS_IMGS_PATHS = {'LEFT': ['media/left_hand_not_detected.png'], 'RIGHT': ['media/right_hand_not_detected.png']}
    
    # Check if there is hand(s) in the image.
    if results.multi_hand_landmarks:
        
        # Iterate over the detected hands in the image.
        for hand_index, hand_info in enumerate(results.multi_handedness):
            
            # Retrieve the label of the hand.
            hand_label = hand_info.classification[0].label
            
            # Update the image path of the hand to a green color hand image.
            # This green image represents that the hand is present in the image. 
            HANDS_IMGS_PATHS[hand_label.upper()] = ['media/'+hand_label.lower()+'_hand_detected.png']
            
            # Check if all the fingers of the hand are up/open.
            if count[hand_label.upper()] == 5:
                
                # Update the image path of the hand to a hand image with green color palm and orange color fingers image.
                # The orange color of a finger represents that the finger is up.
                HANDS_IMGS_PATHS[hand_label.upper()] = ['media/'+hand_label.lower()+'_all_fingers.png']
            
            # Otherwise if all the fingers of the hand are not up/open.
            else:
                
                # Iterate over the fingers statuses of the hands.
                for finger, status in fingers_statuses.items():
                    
                    # Check if the finger is up and belongs to the hand that we are iterating upon.
                    if status == True and finger.split("_")[0] == hand_label.upper():
                        
                        # Append another image of the hand in the list inside the dictionary.
                        # This image only contains the finger we are iterating upon of the hand in orange color.
                        # As the orange color represents that the finger is up.
                        HANDS_IMGS_PATHS[hand_label.upper()].append('media/'+finger.lower()+'.png')
    
    ########################################################################################################################
    
    # Overlay the selected hands prints on the input image.
    ########################################################################################################################
    
    # Iterate over the left and right hand.
    for hand_index, hand_imgs_paths in enumerate(HANDS_IMGS_PATHS.values()):
        
        # Iterate over the images paths of the hand.
        for img_path in hand_imgs_paths:
            
            # Read the image including its alpha channel. The alpha channel (0-255) determine the level of visibility. 
            # In alpha channel, 0 represents the transparent area and 255 represents the visible area.
            hand_imageBGRA = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
            # Retrieve all the alpha channel values of the hand image. 
            alpha_channel = hand_imageBGRA[:,:,-1]
            
            # Retrieve all the blue, green, and red channels values of the hand image.
            # As we also need the three-channel version of the hand image. 
            hand_imageBGR  = hand_imageBGRA[:,:,:-1]
            
            # Retrieve the height and width of the hand image.
            hand_height, hand_width, _ = hand_imageBGR.shape

            # Retrieve the region of interest of the output image where the handprint image will be placed.
            ROI = output_image[30 : 30 + hand_height,
                               (hand_index * width//2) + width//12 : ((hand_index * width//2) + width//12 + hand_width)]
            
            # Overlay the handprint image by updating the pixel values of the ROI of the output image at the 
            # indexes where the alpha channel has the value 255.
            ROI[alpha_channel==255] = hand_imageBGR[alpha_channel==255]

            # Update the ROI of the output image with resultant image pixel values after overlaying the handprint.
            output_image[30 : 30 + hand_height,
                         (hand_index * width//2) + width//12 : ((hand_index * width//2) + width//12 + hand_width)] = ROI
    
    ########################################################################################################################
    
    # Check if the output image is specified to be displayed.
    if display:

        # Display the output image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1])
        plt.title("Output Image")
        plt.axis('off')
    
    # Otherwise
    else:

        # Return the output image
        return output_image




def drawBasicGrid(img, grid_size = (3, 3), color = (124, 25, 45)):

    x_size = img.shape[0]
    y_size = img.shape[1]

    x_line_cords = np.linspace(0+200, x_size-200, grid_size[0]+1).astype(int)
    y_line_cords = np.linspace(0+200, y_size-200, grid_size[1]+1).astype(int)
    for x in x_line_cords:
        img = cv2.line(img, (int(x), 0+200), (int(x), x_size-200), color=color, thickness=10)

    for y in y_line_cords:
        img = cv2.line(img, (0+200, int(y)), (y_size-200, int(y)), color=color, thickness=10)

    return img, (x_line_cords, y_line_cords)



def draw_cursor(img, results):

    cursors = {
                'RIGHT': [None, None],
                'LEFT': [None, None]
            }
    # Iterate over the found hands in the image.
    try:
        for hand_index, hand_info in enumerate(results.multi_handedness):
            
            # Retrieve the label of the found hand.
            hand_label = hand_info.classification[0].label
            
            # Retrieve the landmarks of the found hand.
            hand_landmark =  results.multi_hand_landmarks[hand_index]

            anchor = hand_landmark.landmark[9]
            x_cor = anchor.x
            y_cor = anchor.y

            img = cv2.line(img, (int(x_cor*img.shape[0]), 0), (int(x_cor*img.shape[0]), img.shape[0]), color=(24, 120, 174), thickness=3)
            img = cv2.line(img, (0, int(y_cor*img.shape[1])), (img.shape[1], int(y_cor*img.shape[1])), color=(24, 120, 174), thickness=3)
            cursors[hand_label.upper()] = [int(x_cor*img.shape[0]), int(y_cor*img.shape[1])]
    except:
        pass
    return img, cursors


# def return_effect_cursor(img, results, gestures, color):

#     cursors = {
#                 'RIGHT': [None, None],
#                 'LEFT': [None, None]
#             }
#     # Iterate over the found hands in the image.
#     try:
#         for hand_index, hand_info in enumerate(results.multi_handedness):
            
#             # Retrieve the label of the found hand.
#             hand_label = hand_info.classification[0].label
            
#             # Retrieve the landmarks of the found hand.
#             hand_landmark =  results.multi_hand_landmarks[hand_index]

#             anchor = hand_landmark.landmark[9]
#             x_cor = anchor.x
#             y_cor = anchor.y

#             gesture = gestures[hand_label]
 
#             if gesture == "HOLD SIGN":


            

#             img = cv2.line(img, (int(x_cor*img.shape[0]), 0), (int(x_cor*img.shape[0]), img.shape[0]), color=(24, 120, 174), thickness=3)
#             img = cv2.line(img, (0, int(y_cor*img.shape[1])), (img.shape[1], int(y_cor*img.shape[1])), color=(24, 120, 174), thickness=3)
#             cursors[hand_label.upper()] = [int(x_cor*img.shape[0]), int(y_cor*img.shape[1])]
#     except:
#         pass
#     return img, cursors




def draw_cursor(img, results):

    cursors = {
                'RIGHT': [None, None],
                'LEFT': [None, None]
            }
    # Iterate over the found hands in the image.
    try:
        for hand_index, hand_info in enumerate(results.multi_handedness):
            
            # Retrieve the label of the found hand.
            hand_label = hand_info.classification[0].label
            
            # Retrieve the landmarks of the found hand.
            hand_landmark =  results.multi_hand_landmarks[hand_index]

            anchor = hand_landmark.landmark[9]
            x_cor = anchor.x
            y_cor = anchor.y

            img = cv2.line(img, (int(x_cor*img.shape[0]), 0), (int(x_cor*img.shape[0]), img.shape[0]), color=(24, 120, 174), thickness=3)
            img = cv2.line(img, (0, int(y_cor*img.shape[1])), (img.shape[1], int(y_cor*img.shape[1])), color=(24, 120, 174), thickness=3)
            cursors[hand_label.upper()] = [int(x_cor*img.shape[0]), int(y_cor*img.shape[1])]
    except:
        pass
    return img, cursors



def draw_effect_cursor(img, results):

    cursors = {
                'RIGHT': [None, None],
                'LEFT': [None, None]
            }
    # Iterate over the found hands in the image.
    try:
        for hand_index, hand_info in enumerate(results.multi_handedness):
            
            # Retrieve the label of the found hand.
            hand_label = hand_info.classification[0].label
            
            # Retrieve the landmarks of the found hand.
            hand_landmark =  results.multi_hand_landmarks[hand_index]

            anchor = hand_landmark.landmark[9]
            x_cor = anchor.x
            y_cor = anchor.y

            x_cursor_pixel = int(x_cor*img.shape[0])
            y_cursor_pixel = int(y_cor*img.shape[1])


            img = cv2.line(img, (int(x_cor*img.shape[0]), 0), (int(x_cor*img.shape[0]), img.shape[0]), color=(24, 120, 174), thickness=3)
            img = cv2.line(img, (0, int(y_cor*img.shape[1])), (img.shape[1], int(y_cor*img.shape[1])), color=(24, 120, 174), thickness=3)
            img = cv2.circle(img, (x_cursor_pixel, y_cursor_pixel), 25, (120, 30, 240), -1)
            cursors[hand_label.upper()] = [int(x_cor*img.shape[0]), int(y_cor*img.shape[1])]
    except:
        pass
    return img, cursors






def draw_effect_push(img, grid, cursors, gestures, color):
    overlay = img.copy()

    hands_labels = ['RIGHT', 'LEFT']

    for hand_label in hands_labels:

        cursor = cursors[hand_label]
        gesture = gestures[hand_label]
 
        if gesture == "HOLD SIGN":
            x_index = bisect(grid[0], cursor[0])
            y_index = bisect(grid[1], cursor[1])
            
            if  x_index <= len(grid[0]) - 1 and y_index <= len(grid[1]) - 1:

                img = cv2.circle(img, (cursor[0], cursor[1]), 25, (120, 30, 240), -1)

                alpha = 0.3  # Transparency factor.

                # Following line overlays transparent rectangle over the image
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
            

    return img



def draw_triger_push(img, grid, cursors, gestures, color):
    overlay = img.copy()

    hands_labels = ['RIGHT', 'LEFT']

    for hand_label in hands_labels:

        cursor = cursors[hand_label]
        gesture = gestures[hand_label]
 
        if gesture == "HOLD SIGN":
            x_index = bisect(grid[0], cursor[0])
            y_index = bisect(grid[1], cursor[1])
            
            if  x_index <= len(grid[0]) - 1 and y_index <= len(grid[1]) - 1:
                img = cv2.rectangle(img, (grid[0][x_index-1], grid[1][y_index-1]), (grid[0][x_index], grid[1][y_index]) ,color, -1)
                alpha = 0.4  # Transparency factor.

                # Following line overlays transparent rectangle over the image
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    return img




def draw_sticky_push(img, grid, cursors, gestures, color):
    overlay = img.copy()

    hands_labels = ['RIGHT', 'LEFT']

    for hand_label in hands_labels:

        cursor = cursors[hand_label]
        gesture = gestures[hand_label]
 
        if gesture == "HOLD SIGN":
            x_index = bisect(grid[0], cursor[0])
            y_index = bisect(grid[1], cursor[1])
            
            if  x_index <= len(grid[0]) - 1 and y_index <= len(grid[1]) - 1:
                img = cv2.rectangle(img, (grid[0][x_index-1], grid[1][y_index-1]), (grid[0][x_index], grid[1][y_index]) ,color, -1)
                alpha = 0.4  # Transparency factor.

                # Following line overlays transparent rectangle over the image
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    return img


def draw_sensitive_push(img, grid, cursors, gestures, action_index, color):
    overlay = img.copy()

    hands_labels = ['RIGHT', 'LEFT']

    for hand_label in hands_labels:

        cursor = cursors[hand_label]
        gesture = gestures[hand_label]
 
        if gesture == "HOLD SIGN":
            x_index = bisect(grid[0], cursor[0])
            y_index = bisect(grid[1], cursor[1])
            
            if  x_index <= len(grid[0]) - 1 and y_index <= len(grid[1]) - 1:
                img = cv2.rectangle(img, (grid[0][x_index-1], grid[1][y_index-1]), (grid[0][x_index], grid[1][y_index]) ,color, -1)
                alpha = 1- action_index[hand_label][3]  # Transparency factor.
                
                if alpha < 0.3:
                    alpha = 0.3
                if alpha > 0.7:
                    alpha = 0.7

                # Following line overlays transparent rectangle over the image
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    return img


def draw_mixing_push(img, grid, cursors, gestures, action_index, color):
    overlay = img.copy()
    
    x_size = img.shape[0]
    y_size = img.shape[1]

    hands_labels = ['RIGHT', 'LEFT']

    for hand_label in hands_labels:

        print("!"*100)
        print(action_index)
        print(grid)

        cursor = cursors[hand_label]
        gesture = gestures[hand_label]
 
        if gesture == "HOLD SIGN":
            x_index = bisect(grid[0], cursor[0])
            y_index = bisect(grid[1], cursor[1])
            
            if  x_index <= len(grid[0]) - 1 and y_index <= len(grid[1]) - 1:
                img = cv2.rectangle(img, (grid[0][x_index-1], int((1-action_index[hand_label][3])* (grid[1][y_index] - grid[1][y_index-1]))), (grid[0][x_index], grid[1][y_index]) ,color, -1)
                alpha = 0.4  # Transparency factor.
                

                # Following line overlays transparent rectangle over the image
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    return img


# def draw_effect_push(img, grid, cursors, gestures, action_index, color):
#     overlay = img.copy()
    
#     x_size = img.shape[0]
#     y_size = img.shape[1]

#     hands_labels = ['RIGHT', 'LEFT']

#     for hand_label in hands_labels:

#         print("!"*100)
#         print(action_index)
#         print(grid)

#         cursor = cursors[hand_label]
#         gesture = gestures[hand_label]
 
#         if gesture != "HOLD SIGN":
#             x_index = bisect(grid[0], cursor[0])
#             y_index = bisect(grid[1], cursor[1])
            
#             if  x_index <= len(grid[0]) - 1 and y_index <= len(grid[1]) - 1:
#                 img = cv2.rectangle(img, (grid[0][x_index-1], int((1-action_index[hand_label][3])* (grid[1][y_index] - grid[1][y_index-1]))), (grid[0][x_index], grid[1][y_index]) ,color, -1)
#                 alpha = 0.4  # Transparency factor.
                

#                 # Following line overlays transparent rectangle over the image
#                 img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

#     return img





def draw_keys(img, grid, page, draw_notes=True):

    counter = 21 + (len(grid[0])-1)*(len(grid[1])-1)*(page-1)
    notes = []
    
    grid_note_lookup = {}

    for y in grid[1][:-1][::-1]:
        for x in grid[0][:-1]:
            if draw_notes:
                cv2.putText(img, str(note_lookup.loc[counter]["NOTE"]) , (x+5, y+50),
                                cv2.FONT_HERSHEY_PLAIN, 4, (0,255,0), 2)
            else:
                cv2.putText(img, str(counter-20) , (x+5, y+50),
                                cv2.FONT_HERSHEY_PLAIN, 4, (0,255,0), 2)
                
            notes.append(counter)
            counter+=1
            grid_note_lookup[counter] = (x, y)

    return img, notes




def draw_effect_points(img, notes, grid, effects_cache):

    grid[0]
    grid[1]

    for effect in notes:
        temp_cache = effects_cache[effect]

        cv2.circle(img, (temp_cache[0], temp_cache[1]), 200, (0,0, 240), -1)