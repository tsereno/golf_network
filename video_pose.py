#!nvidia-smi
#!pip install opencv-python
#https://google.github.io/mediapipe/getting_started/gpu_support.html
#!pip install mediapipe

import pandas as pd
import numpy as np
import os
import time
import csv
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
#mp_holistic = mp.solutions.holistic

directory = './'
'''
    For the given path, get the List of all files in the directory tree 
'''
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

# Get the list of all files in directory tree at given path
list_of_files = getListOfFiles(directory)

#list_of_files = sorted( filter( lambda x: os.path.isfile(os.path.join(directory, x)),
#                        os.listdir(directory) ) )

for entry in list_of_files:
 if (entry.endswith('.MTS') or entry.endswith('.MOV')) and not os.path.exists(entry+'.csv'):
    frame_number = 0
    count = 0
    alldata = []
    fps_time = 0
    subdirname = os.path.basename(os.path.dirname(entry))
    #label = int(subdirname)
    label = 1
    pose_tubuh = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
                  'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB',
                  'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']

    #pose_tangan = ['WRIST', 'THUMB_CPC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP',
    #               'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
    #               'RING_FINGER_MCP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']

    #pose_tangan_2 = ['WRIST2', 'THUMB_CPC2', 'THUMB_MCP2', 'THUMB_IP2', 'THUMB_TIP2', 'INDEX_FINGER_MCP2', 'INDEX_FINGER_PIP2', 'INDEX_FINGER_DIP2', 'INDEX_FINGER_TIP2', 'MIDDLE_FINGER_MCP2',
    #               'MIDDLE_FINGER_PIP2', 'MIDDLE_FINGER_DIP2', 'MIDDLE_FINGER_TIP2', 'RING_FINGER_PIP2', 'RING_FINGER_DIP2', 'RING_FINGER_TIP2',
    #               'RING_FINGER_MCP2', 'PINKY_MCP2', 'PINKY_PIP2', 'PINKY_DIP2', 'PINKY_TIP2']

    '''
    #video_cap = cv2.VideoCapture(0)
    video_cap = cv2.VideoCapture("00014.MTS", cv2.CAP_FFMPEG)

    # returns the frame rate
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    print("Frame rate: ", int(fps), "FPS")

    while True:
        # `success` is a boolean and `frame` contains the next video frame
        success, frame = video_cap.read()
        cv2.imshow("frame", frame)
        # wait 20 milliseconds between frames and break the loop if the `q` key is pressed
        if cv2.waitKey(20) == ord('q'):
            break

    # we also need to close the video and destroy all Windows
    video_cap.release()
    cv2.destroyAllWindows()
    '''

    # For webcam input:
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(entry, cv2.CAP_FFMPEG)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frame rate: ", int(fps), "FPS")


    #with mp_holistic.Holistic(
    with mp_pose.Pose(
        min_detection_confidence=0.7,
        #model complexity 0=low, 1=medium, 2=heavy
        min_tracking_confidence=0.7, model_complexity=0,static_image_mode=False,smooth_landmarks=True) as pose:
        #min_tracking_confidence=0.5) as holistic:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          break
          #continue
        frame_number+=1
        if(frame_number % 3 == 0):
         if int(subdirname)==0:
          time.sleep(.05)

         # To improve performance, optionally mark the image as not writeable to
         # pass by reference.
         image.flags.writeable = False
         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
         results = pose.process(image)
         #results = holistic.process(image)

         # Draw the pose annotation on the image.
         image.flags.writeable = True
         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
         mp_drawing.draw_landmarks(
             image,
             results.pose_landmarks,
             mp_pose.POSE_CONNECTIONS,
             landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

         '''
         # Draw landmark annotation on the image.
         image.flags.writeable = True
         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
         mp_drawing.draw_landmarks(
             image,
             results.face_landmarks,
             mp_holistic.FACEMESH_CONTOURS,
             landmark_drawing_spec=None,
             connection_drawing_spec=mp_drawing_styles
             .get_default_face_mesh_contours_style())
         mp_drawing.draw_landmarks(
             image,
             results.pose_landmarks,
             mp_holistic.POSE_CONNECTIONS,
             landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())
         '''

         # Save landmarks.
         if results.pose_landmarks is not None:
             # Check the number of landmarks and take pose landmarks.
             assert len(results.pose_landmarks.landmark) == 33, 'Unexpected number of predicted pose landmarks: {}'.format(len(results.pose_landmarks.landmark))
             #pose_landmarks = [[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in results.pose_landmarks.landmark]
             pose_landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]

             # Map pose landmarks from [0, 1] range to absolute coordinates to get
             # correct aspect ratio.
             #frame_height, frame_width = image.shape[:2]
             #pose_landmarks *= np.array([frame_width, frame_height, frame_width])

             # Write pose sample to CSV.
             pose_landmarks = np.around(pose_landmarks, 5).flatten().astype(str).tolist()
             pose_landmarks = pose_landmarks + [label]
             alldata.append(pose_landmarks)

         # use this break statement to check your data before processing the whole video
         #if frame_number == 600: break
         #print(frame_number)

         # Flip the image horizontally for a selfie-view display.
         #cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))

         if int(subdirname)==0:
          cv2.imshow('MediaPipe Pose', image)
         #if cv2.waitKey(5) & 0xFF == ord('0'):
         #  label = 0
         key = cv2.waitKey(1)
         #print(key)
         if key == 27:
            break
         if key == 49:
           if label == 0:
            label = 1
           else:
            label = 0
         #if cv2.waitKey(5) & 0xFF == 27:
         #  break
    cap.release()

    # write the data to a .csv file
    outfile_path = entry+'.csv'
    df = pd.DataFrame(alldata)
    df = df.rename(columns={99: "label"})
    df.to_csv(outfile_path, index = False)
    print('save complete')
    
    
    