import pandas as pd
import numpy as np
import os
import sys

pose_tubuh = ['NOSE_X',
              'NOSE_Y',
              'NOSE_Z',
              'LEFT_EYE_INNER_X', 
              'LEFT_EYE_INNER_Y', 
              'LEFT_EYE_INNER_Z', 
              'LEFT_EYE_X', 
              'LEFT_EYE_Y', 
              'LEFT_EYE_Z', 
              'LEFT_EYE_OUTER_X', 
              'LEFT_EYE_OUTER_Y', 
              'LEFT_EYE_OUTER_Z', 
              'RIGHT_EYE_INNER_X', 
              'RIGHT_EYE_INNER_Y', 
              'RIGHT_EYE_INNER_Z', 
              'RIGHT_EYE_X', 
              'RIGHT_EYE_Y', 
              'RIGHT_EYE_Z', 
              'RIGHT_EYE_OUTER_X', 
              'RIGHT_EYE_OUTER_Y', 
              'RIGHT_EYE_OUTER_Z', 
              'LEFT_EAR_X', 
              'LEFT_EAR_Y', 
              'LEFT_EAR_Z', 
              'RIGHT_EAR_X', 
              'RIGHT_EAR_Y', 
              'RIGHT_EAR_Z', 
              'MOUTH_LEFT_X', 
              'MOUTH_LEFT_Y', 
              'MOUTH_LEFT_Z', 
              'MOUTH_RIGHT_X',
              'MOUTH_RIGHT_Y',
              'MOUTH_RIGHT_Z',
              'LEFT_SHOULDER_X',
              'LEFT_SHOULDER_Y',
              'LEFT_SHOULDER_Z',
              'RIGHT_SHOULDER_X', 
              'RIGHT_SHOULDER_Y', 
              'RIGHT_SHOULDER_Z', 
              'LEFT_ELBOW_X', 
              'LEFT_ELBOW_Y', 
              'LEFT_ELBOW_Z', 
              'RIGHT_ELBOW_X', 
              'RIGHT_ELBOW_Y', 
              'RIGHT_ELBOW_Z', 
              'LEFT_WRIST_X', 
              'LEFT_WRIST_Y', 
              'LEFT_WRIST_Z', 
              'RIGHT_WRIST_X', 
              'RIGHT_WRIST_Y', 
              'RIGHT_WRIST_Z', 
              'LEFT_PINKY_X', 
              'LEFT_PINKY_Y', 
              'LEFT_PINKY_Z', 
              'RIGHT_PINKY_X', 
              'RIGHT_PINKY_Y', 
              'RIGHT_PINKY_Z', 
              'LEFT_INDEX_X', 
              'LEFT_INDEX_Y', 
              'LEFT_INDEX_Z', 
              'RIGHT_INDEX_X', 
              'RIGHT_INDEX_Y', 
              'RIGHT_INDEX_Z', 
              'LEFT_THUMB_X',
              'LEFT_THUMB_Y',
              'LEFT_THUMB_Z',
              'RIGHT_THUMB_X', 
              'RIGHT_THUMB_Y', 
              'RIGHT_THUMB_Z', 
              'LEFT_HIP_X', 
              'LEFT_HIP_Y', 
              'LEFT_HIP_Z', 
              'RIGHT_HIP_X', 
              'RIGHT_HIP_Y', 
              'RIGHT_HIP_Z', 
              'LEFT_KNEE_X', 
              'LEFT_KNEE_Y', 
              'LEFT_KNEE_Z_Z', 
              'RIGHT_KNEE_X', 
              'RIGHT_KNEE_Y', 
              'RIGHT_KNEE_Z', 
              'LEFT_ANKLE_X', 
              'LEFT_ANKLE_Y', 
              'LEFT_ANKLE_Z', 
              'RIGHT_ANKLE_X', 
              'RIGHT_ANKLE_Y', 
              'RIGHT_ANKLE_Z', 
              'LEFT_HEEL_X', 
              'LEFT_HEEL_Y', 
              'LEFT_HEEL_Z', 
              'RIGHT_HEEL_X', 
              'RIGHT_HEEL_Y', 
              'RIGHT_HEEL_Z', 
              'LEFT_FOOT_INDEX_X', 
              'LEFT_FOOT_INDEX_Y', 
              'LEFT_FOOT_INDEX_Z', 
              'RIGHT_FOOT_INDEX_X',
              'RIGHT_FOOT_INDEX_Y',
              'RIGHT_FOOT_INDEX_Z',
              'ball1',
              'ball2',
              'ball3',
              'ball4',
              'ball5',
              'flight1',
              'flight2',
              'flight3',
              'flight4',
              'flight5',
              'flight6',
              'flight7',
              'club1',
              'club2',
              'frame_number',
			  'label',
              'graph_id',
              'filename',
              'name',
              'club']

directory = './'+sys.argv[1]+"/"

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

all_data = pd.DataFrame()
swingID = 0

for entry in list_of_files:
 if entry.endswith('.csv'):
  #print(entry)
  subdirname = os.path.basename(os.path.dirname(entry))
  #print(subdirname)
  mainfolder=os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(entry))))
  #print(mainfolder)
  split = mainfolder.split('_', 2)
  name = split[0]
  club = split[1]
  #print(name,club)
  temp_data = pd.read_csv(entry)
  #temp_data = temp_data.head(1)
  temp_data['swing_id']=swingID
  temp_data['filename']=entry
  temp_data['name']=name
  temp_data['club']=club
  #temp_data = temp_data.assign(row_number=range(len(temp_data)))
  #print(temp_data)
  all_data = pd.concat([all_data, temp_data], axis="rows", ignore_index=True)
  swingID += 1
all_data.columns = pose_tubuh
#print(all_data)
all_data.to_csv('golfswings.csv', index = False)
