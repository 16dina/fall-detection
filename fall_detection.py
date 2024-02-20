# IMPORT NECESSARY LIBRARIES
from ultralytics import YOLO
import requests
from azure.storage.blob import BlobServiceClient, ContentSettings
import json
import cv2
import numpy as np
import tempfile
import time
import requests
from copy import deepcopy

# DEFINING THE POSE MODEL FROM YOLOv8
model = YOLO('yolo models/yolov8n-pose.pt')

## CONNECTING TO AZURE BLOB STORAGE:
# DEFINE AZURE STORAGE CONFIGURATIONS
AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=stproject4tm20241;AccountKey=f71FXw0s7LEAUAeTJn/S4iEOJSJ3QNrmCFznzdOnUXNhNHtPQ1ePgdtll1ouTCp0+7sYj7oQlv6/+ASt49Bm/A==;EndpointSuffix=core.windows.net"
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
VIDEO_FPS = 10
fall_alerted = False
video_frames_before = []
frozen_video_frames_before = []
video_frames_after = []
taking_video = False
clip_frames = []

# GETTING THE Y VALUES OF THE PERSON
def frame_coordinates(frame):
    y_values_frame = [keypoint[1].numpy() for keypoint in frame.keypoints.xy[0] if keypoint[1].numpy() != 0]
    return y_values_frame

# DEFINING THE STARTING THRESHOLDS (BOTH FALLING THRESHOLD AND THRESHOLDS FOR LAYING DOWN)
def get_starting_frames(results):
    global first_point_threshold
    global second_point_threshold
    global falling_threshold

    start_time = time.time()

    first_frame = next(results, None)
    if first_frame is not None:
        y_values_first_frame = frame_coordinates(first_frame)
        while(len(y_values_first_frame) < 6):
            first_frame = next(results, None)
            y_values_first_frame = frame_coordinates(first_frame)
        falling_threshold = ((y_values_first_frame[len(y_values_first_frame)-1] - y_values_first_frame[0]) * 2/3) + 20 # Buffer (can be changed)
            
    print("Falling threshold:", falling_threshold)

    second_frame = next(results, None)
    if second_frame is not None:
        y_values_second_frame = frame_coordinates(second_frame)
        while(len(y_values_second_frame) < 6):
            second_frame = next(results, None)
            y_values_second_frame = frame_coordinates(second_frame)
    
    first_point_diff = abs(y_values_first_frame[0] - y_values_second_frame[0])
    second_point_diff = abs(y_values_first_frame[5] - y_values_second_frame[5])
    first_point_threshold = first_point_diff + 15 # Buffer (can be changed)
    second_point_threshold = second_point_diff + 15 # Buffer (can be changed)

    print("First point threshold:", first_point_threshold)
    print("Second point threshold:", second_point_threshold)

    return first_point_threshold, second_point_threshold, start_time

# CHECK FALLING (WILL REFER TO CHECKING THE FALLING TIME TOO)
def check_falling(y_values):
    global previous_y_values
    global fallen_state
    global minimum
    global maximum
    global fall_start_time
    global elapsed_time_states
    global taking_video
    global fall_alerted
    global video_frames_after

    # This applies checking if it's a fall or laying down
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
                    taking_video = True
                    fall_start_time = time.time()
                    elapsed_time_states.append("Fallen")
                    print("Fallen")
                    print("STARTING TIME OF FALL:", fall_start_time)
                    print("states:", elapsed_time_states)
        else:
            # If they are standing but 10 seconds have passed already after fall then we still want to see it
            # This handles it to keep taking the video
            if(fall_alerted):                
                taking_video = True
            else:
                # If they're just safe and stood up before the 10 seconds, then reset
                fallen_state = False
                taking_video = False
                frozen_video_frames_before.clear()
                video_frames_after.clear()
                
            fall_start_time = None
            elapsed_time_states.clear()
            print("Safe")

    # Update previous frame's values
    previous_y_values = y_values


# IS THE FALL 10 OR MORE SECONDS LONG? ==> HERE WE DETERMINE IF IT COUNTS AS A FALL OR NOT
def check_falling_time(fall_start_time, elapsed_time_states):
    global fallen_state
    global taking_video
    global fall_alerted
    global duration_of_fall
    # Perform subtraction only if fall_start_time is not None
    if fall_start_time is not None:
        duration_of_fall = time.time() - fall_start_time
        print("Duration of fall:", duration_of_fall)
        if duration_of_fall >= MIN_ELAPSED_TIME_THRESHOLD:
            print("Elapsed time states:", elapsed_time_states)
            print("FALL ALERT!!!")
            fall_alerted = True
            taking_video = True
            fall_start_time = None
            elapsed_time_states.clear()
            fallen_state = False

    return fall_start_time, elapsed_time_states

# This is the same as check_falling_time but it's specific for cases when the person is no longer in the frame (no detection)
# In this case the YOLO model doesn't have any results and thus the OpenCV stops capturing frames
# To solve this issue, we call this function when 'No human detected' which saves the same last captured frame until it forms the required length
# Then we call the functions to save the video clip and alert
# Note: There is no real reason to capture more of different frames when the person is completely out of the frame
#       The only necessity is to check whether they're still outside of the frame (not detected for occluded falls detection)
#       If the person re-appears in the frame, the other function check_falling_time is called
def check_falling_time_out_of_frame(fall_start_time, elapsed_time_states):
    global fallen_state
    global taking_video
    global fall_alerted
    global duration_of_fall
    if fall_start_time is not None:
        # Perform subtraction only if fall_start_time is not None
        duration_of_fall = time.time() - fall_start_time
        print("Duration of fall:", duration_of_fall)
        if duration_of_fall >= MIN_ELAPSED_TIME_THRESHOLD:
            print("Elapsed time states:", elapsed_time_states)
            print("FALL ALERT!!!")
            while len(video_frames_after) <= 150:
                video_frames_after.append(r.orig_img)
            save_video_clip()
            save_info_in_blob()
            send_api_call()
            taking_video = False
            video_frames_before.clear()
            video_frames_after.clear()
            frozen_video_frames_before.clear()
            fall_start_time = None
            elapsed_time_states.clear()
            fallen_state = False
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
    global clip_frames
    clip_frames = frozen_video_frames_before + video_frames_after

    if not clip_frames:
        print("No frames to save.")
        return

    # Verify individual frames (optional)
    for frame in clip_frames:
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame detected in clip_frames.")

    # Save the clip to a temporary file
    temp_file_path = tempfile.mktemp(suffix=".mp4")
    print(temp_file_path)
    out = cv2.VideoWriter(temp_file_path, cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, (frame.shape[1], frame.shape[0]))
    
    for frame in clip_frames:
        out.write(frame)

    out.release()

    # Upload the clip to Azure Blob Storage
    upload_clip_to_blob(temp_file_path)
    print("File path:", temp_file_path)

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
    global fall_alerted
    # Set the request url
    url = 'https://appserviceproject4tm20241.azurewebsites.net/api/visiage/fallen'
    # Set the request headers
    # The camera ID is to be setup during installation manually
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
    fall_alerted = False



### START OF MAIN CODE PART ###
        
# PROCESS THE FRAME USING YOLO ==> APPLYING KEYPOINTS AND HUMAN DETECTION
results = model(source=0, show=True, conf=0.3, stream=True, save=True)

# EXTRACT KEYPOINTS' COORDINATES AND PERFORM FURTHER ANALYSIS
if results:
    first_point_threshold, second_point_threshold, start_time = get_starting_frames(results)
    for r in results:
        # Sliding window technique where we update the thresholds every 5 seconds
        if time.time() - start_time > 5 and fallen_state == False:
            first_point_threshold, second_point_threshold, start_time = get_starting_frames(results)
        # Update dynamic list with the latest frame
        print("Length of BEFORE:", len(video_frames_before))
        print("Length of FROZEN:", len(frozen_video_frames_before))
        print("Length of AFTER:", len(video_frames_after))
        print("Fall state:", fallen_state)
        print("Fall alerted:", fall_alerted)
        print("Taking video:", taking_video)

        # Add frames to start frames (before falling). This is continuous
        if len(video_frames_before) > 100:
            video_frames_before.pop(0)
        else:
            video_frames_before.append(r.orig_img) 

        # When a fall happens, taken video is true
        if taking_video:
            # Freeze before frames
            if(len(frozen_video_frames_before) == 0):
                frozen_video_frames_before = deepcopy(video_frames_before)
            # Add to after frames
            # If it's not 10 seconds after falling yet then keep adding frames
            if len(video_frames_after) <= 150:
                video_frames_after.append(r.orig_img)
            else:
                # When it's 10 seconds save video and send alert
                save_video_clip()
                save_info_in_blob()
                send_api_call()
                taking_video = False
                video_frames_before.clear()
                video_frames_after.clear()
                frozen_video_frames_before.clear()
                fall_start_time = None
                elapsed_time_states.clear()
                fallen_state = False
        else:
            # If not taking video then update the frames before
            if len(video_frames_before) > 100:
                video_frames_before.pop(0)
            else:
                video_frames_before.append(r.orig_img) 

        y_values = frame_coordinates(r)
        # If 6 keypoints are showing then check for falling
        if len(y_values) >= 6:
            minimum = min(y_values)
            maximum = max(y_values)
            check_falling(y_values)
        else:
            # If less than 6 keypoints then check if they fell (No human is detected in this case)
            if fallen_state == True:
                elapsed_time_states.append("No human detected")
                fall_start_time, elapsed_time_states = check_falling_time_out_of_frame(fall_start_time, elapsed_time_states)
                print("No human detected.")
                print("states:", elapsed_time_states)

        cv2.imshow("Video Feed", r.orig_img)