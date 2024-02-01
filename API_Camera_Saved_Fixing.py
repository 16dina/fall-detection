# Send to blob storage when camera detects falling for more than x seconds
# No need for fastapi 

# IMPORT NECESSARY LIBRARIES
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from typing import List
import requests
from azure.storage.blob import BlobServiceClient, ContentSettings
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
import shutil
import cv2
import numpy as np
import tempfile
import os
import time
import requests
import asyncio

# DEFINING THE POSE MODEL FROM YOLO
model = YOLO('yolo models/yolov8n-pose.pt')

## CONNECTING TO AZURE BLOB STORAGE:
# DEFINE AZURE STORAGE CONFIGURATIONS
AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=stproject4tm20241;AccountKey=y+3jml6m4c4bMQcgUd87MeP9rfUDaJqfYKBznSqbzFn10J6OV3pnX4fzJxDC+WG2H/h2ultIIPf4+AStaBffIA==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "mycontainer"
# CREATE BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

# DEFINING VARIABLES AND CONSTANTS FOR FALLING/LAYING DOWN
previous_y_values = None # previous frame y values to compare
first_point_threshold = None # difference in distance between keypoint 0
second_point_threshold = None # difference in distance between keypoint 5
falling_threshold = None # falling threshold
fallen_state = False # fallen state identifier
blob_number = 1 # blob number to be used for blob name
# DEFINE THE MINIMUM TIME REQUIRED TO TRIGGER A FALL
MIN_ELAPSED_TIME_THRESHOLD = 10
fall_start_time = None
elapsed_time_states = []
# VIDEO SAVING
video_buffer = []
VIDEO_FPS = 10

# GETTING THE Y VALUES OF THE PERSON
def frame_coordinates(frame):
    y_values_frame = [keypoint[1].numpy() for keypoint in frame.keypoints.xy[0] if keypoint[1].numpy() != 0]
    return y_values_frame

# DEFINING THE STARTING THRESHOLDS (BOTH FALLING THRESHOLD AND THRESHOLDS FOR LAYING DOWN)
def get_starting_frames(results):
    global first_point_threshold
    global second_point_threshold
    global falling_threshold

    first_frame = next(results, None)
    if first_frame is not None:
        y_values_first_frame = frame_coordinates(first_frame)
        while(len(y_values_first_frame) < 9): # used to be 9
            first_frame = next(results, None)
            y_values_first_frame = frame_coordinates(first_frame)
        falling_threshold = ((y_values_first_frame[len(y_values_first_frame)-1] - y_values_first_frame[0]) * 2/3) + 20
            
    print("Falling threshold:", falling_threshold)

    second_frame = next(results, None)
    if second_frame is not None:
        y_values_second_frame = frame_coordinates(second_frame)
        while(len(y_values_second_frame) < 9): # used to be 9
            second_frame = next(results, None)
            y_values_second_frame = frame_coordinates(second_frame)
    
    first_point_diff = abs(y_values_first_frame[0] - y_values_second_frame[0])
    second_point_diff = abs(y_values_first_frame[5] - y_values_second_frame[5])
    first_point_threshold = first_point_diff + 15
    second_point_threshold = second_point_diff + 15

    print("First point threshold:", first_point_threshold)
    print("Second point threshold:", second_point_threshold)

    return first_point_threshold, second_point_threshold

# CHECK FALLING (WILL REFER TO CHECKING THE FALLING TIME TOO)
def check_falling(y_values):
    global previous_y_values
    global fallen_state
    global minimum
    global maximum
    global fall_start_time
    global elapsed_time_states

    if previous_y_values is not None and len(y_values) >= 6 and len(previous_y_values) >= 6:
        first_point_diff = abs(previous_y_values[0] - y_values[0])
        second_point_diff = abs(previous_y_values[5] - y_values[5])

        if (falling_threshold is not None) and (maximum - minimum <= falling_threshold):
            if (first_point_diff <= first_point_threshold) and (second_point_diff <= second_point_threshold):
                print("Laying down")
                if fallen_state:
                    elapsed_time_states.append("Laying down")
                    fall_start_time, elapsed_time_states = check_falling_time(fall_start_time, elapsed_time_states)
                    print("states:", elapsed_time_states)
                else:
                    fall_start_time = None
            else:
                if fallen_state:
                    elapsed_time_states.append("Fallen")
                    fall_start_time, elapsed_time_states = check_falling_time(fall_start_time, elapsed_time_states)
                    print("Fallen")
                    print("states:", elapsed_time_states)
                else:
                    fallen_state = True
                    fall_start_time = time.time()
                    elapsed_time_states.append("Fallen")
                    print("Fallen")
                    print("STARTING TIME OF FALL:", fall_start_time)
                    print("states:", elapsed_time_states)
        else:
            fall_start_time = None
            elapsed_time_states.clear()
            fallen_state = False
            print("Safe")

    #fall_start_time, elapsed_time_states = check_falling_time(fall_start_time, elapsed_time_states)
    previous_y_values = y_values

# IS THE FALL 10 OR MORE SECONDS LONG? ==> HERE WE DETERMINE IF IT COUNTS AS A FALL OR NOT
def check_falling_time(fall_start_time, elapsed_time_states):
    global fallen_state
    global video_buffer
    if fall_start_time is not None:
        # Perform subtraction only if fall_start_time is not None
        duration_of_fall = time.time() - fall_start_time
        print("Duration of fall:", duration_of_fall)
        if duration_of_fall >= MIN_ELAPSED_TIME_THRESHOLD:
            print("Elapsed time states:", elapsed_time_states)
            save_video_clip()
            save_info_in_blob()
            send_api_call()
            fall_start_time = None
            elapsed_time_states.clear()
            fallen_state = False
            video_buffer = []
    return fall_start_time, elapsed_time_states

# UPLOADING THE CLIP (TO BE DEFINED IN THE NEXT FUNCTION) TO AZURE BLOB STORAGE
def upload_clip_to_blob(clip_path):
    global video_blob_name
    try:
        # Read the video clip content
        with open(clip_path, "rb") as clip_file:
            clip_content = clip_file.read()

        # Set the blob name and upload the clip to Azure Blob Storage
        video_blob_name = "fallen_clip_" + str(time.time()) + ".mp4"
        blob_client = container_client.get_blob_client(video_blob_name)
        blob_client.upload_blob(clip_content, overwrite=True, content_settings=ContentSettings(content_type="video/mp4"))

        print("Fallen clip uploaded to Azure Blob Storage")
    except Exception as e:
        print(f"Error uploading fallen clip to Azure Blob Storage: {str(e)}")

# SAVING A VIDEO CLIP OF THE FALL TO TEMPORARY STORAGE ON PC (EDGE HARDWARE)
def save_video_clip():
    try:
        clip_duration_before = 10  # seconds before the fall
        clip_duration_after = 30   # seconds after the fall (you can adjust this)

        fall_index = len(video_buffer) - 1
        fall_index = max(clip_duration_before * VIDEO_FPS, fall_index)
        fall_index = min(len(video_buffer) - clip_duration_after * VIDEO_FPS, fall_index)

        start_index = fall_index - clip_duration_before * VIDEO_FPS
        # end_index = fall_index + clip_duration_after * VIDEO_FPS
        end_index = min(len(video_buffer), fall_index + clip_duration_after * VIDEO_FPS)  # Fix here


        clip_frames = video_buffer[start_index:end_index]

        # Verify individual frames (optional)
        for frame in clip_frames:
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                raise ValueError("Invalid frame detected in clip_frames.")

        # Save the clip to a temporary file
        temp_file_path = tempfile.mktemp(suffix=".mp4")
        out = cv2.VideoWriter(temp_file_path, cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, (frame.shape[1], frame.shape[0]))
        
        for frame in clip_frames:
            out.write(frame)

        out.release()

        # Upload the clip to Azure Blob Storage
        upload_clip_to_blob(temp_file_path)
        print("File path:", temp_file_path)

    except Exception as e:
        print(f"Error saving video clip: {str(e)}")
        # Add additional error handling or logging as needed

# SAVING INFORMATION (STATE LABEL, TIMESTAMP, FALLING VIDEO BLOB NAME) TO AZURE AS JSON
def save_info_in_blob():
    global blob_number
    global blob_name
    try:
        fallen_info = {"status": "Fallen", "timestamp": str(time.time()), "filename": video_blob_name}
        fall_info_json = json.dumps(fallen_info)

        blob_name = "incident_" + str(blob_number)
        blob_number += 1

        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(fall_info_json, overwrite=True)

        print("Fall detection information uploaded to Azure Blob Storage")
    except Exception as e:
        print(f"Error uploading fall detection information to Azure Blob Storage: {str(e)}")

# FALLEN POST REQUEST TO APP'S API ENDPOINT ==> TRIGGERS AN ALERT ON THE APP
def send_api_call():
    # Set the request url
    url = 'https://appserviceproject4tm20241.azurewebsites.net/api/visiage/fallen'
    # Set the request headers
    data = {"blobFileName": blob_name, "videoBlobFile": video_blob_name, "cameraId": "1" }
    # Make a POST request
    response = requests.post(url, json=data)
    # Check if the request was successful (status code 200)
    if response.status_code == 200 or response.status_code == 201:
        print('POST request successful')
        print('Response:', response.text)
    else:
        print(f'Error: {response.status_code}')
        print('Response:', response.text)



### START OF MAIN CODE PART ###
        
# PROCESS THE FRAME USING YOLO ==> APPLYING KEYPOINTS AND HUMAN DETECTION
results = model(source=0, show=True, conf=0.5, stream=True)

# EXTRACT KEYPOINTS' COORDINATES AND PERFORM FURTHER ANALYSIS
if results:
    first_point_threshold, second_point_threshold = get_starting_frames(results)
    for r in results:
        y_values = frame_coordinates(r)
        # for y in y_values:
        #     print(y)
        if len(y_values) >= 6:
            minimum = min(y_values)
            maximum = max(y_values)
            #print("minimum y value is:", minimum)
            #print("maximum y value is:", maximum)
            check_falling(y_values)
        else:
            if fallen_state == True:
                elapsed_time_states.append("No human detected")
                fall_start_time, elapsed_time_states = check_falling_time(fall_start_time, elapsed_time_states)
                print("No human detected.")
                #print("time:", fall_start_time)
                print("states:", elapsed_time_states)
            else:
                # here...
                #print("diff", maximum-minimum)
                #print("start time:", fall_start_time)
                print("states:", elapsed_time_states)
                print("No human detected.")

        frame_np = r.orig_img
        video_buffer.append(frame_np)
        cv2.imshow("Video Feed", frame_np)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break