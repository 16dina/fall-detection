# The setup and initialization of this code requires that the elder is standing alone in front of the camera at the moment of setup
# This will help get the correct coordinates to base incidents on

# The logic is as follows:

# If the distance between the highest and lowest keypoint is less than a threshold:

#       Is the difference between keypoint 0 in the current and previous frame big? (same for keypoint 4 to double check) 
#               --> Falling (Because it means they descended suddenly, so fell).

#       Is the difference small? 
#               --> Laying down (Because it means the descended slowly, so on purpose).


# If the distance between the highest and lowest is more or equal to the threshold --> Standing (Safe).


# Suggestion: 10 frames sliding window
# thresholds should continuously change based on the first from the 10
# camera placemnet is important
# for the obscured: check if only a few are showing and how fast they disappeared (the nose one) 
# --> velocity calculation uses distance and time. ask chatgpt
# focus now on fast api and putting everything to a docker container
# send data automatically from webcam to the storage in azure
# then figure out how to make api calls continuously to the storage and analyze the data then only do a post call when we find a fall

# Importing YOLO library
from ultralytics import YOLO

# Defining the pose model from YOLO
model = YOLO('yolo models/yolov8n-pose.pt')
# Saving the results from a source (a video or camera)
results = model(source=0, show=True, conf=0.3, stream=True)

# Defining some global variables:
previous_y_values = None
first_point_diff = 0
second_point_diff = 0 
first_point_threshold = None
second_point_threshold = None
falling_threshold = None
fallen_state = False

# Function to get coordinates of keypoints on a frame
def frame_coordinates(frame):
    y_values_frame = [keypoint[1].numpy() for keypoint in frame.keypoints.xy[0] if keypoint[1].numpy() != 0]
    return y_values_frame

# Function to check falling or laying down or standing (safe)
def check_falling(y_values):
    global previous_y_values
    global fallen_state

    if previous_y_values is not None:
        first_point_diff = abs(previous_y_values[0] - y_values[0])
        print("first point difference:", first_point_diff)
        second_point_diff = abs(previous_y_values[5] - y_values[5])
        print("second point difference:", second_point_diff)

        if (falling_threshold is not None) and (maximum - minimum <= falling_threshold):
            if (first_point_diff <= first_point_threshold) and (second_point_diff <= second_point_threshold):
                print("Laying down")
            else:
                fallen_state = True
                print("Fallen")                
        else:
            fallen_state = False
            print("Safe")

    previous_y_values = y_values


def get_starting_frames(results):
    # Global Variables
    global first_point_threshold
    global second_point_threshold
    global falling_threshold

    # First Frame
    first_frame = next(results, None)
    if first_frame is not None:
        y_values_first_frame = frame_coordinates(first_frame)
        while(len(y_values_first_frame) < 9): # While we don't have enough keypoints check again
            first_frame = next(results, None)
            y_values_first_frame = frame_coordinates(first_frame)
        falling_threshold = ((y_values_first_frame[len(y_values_first_frame)-1] - y_values_first_frame[0]) * 2/3) + 20 # callibration: relative
            
    print("Falling threshold:", falling_threshold)

    # Second Frame
    second_frame = next(results, None)
    if second_frame is not None:
        y_values_second_frame = frame_coordinates(second_frame)
        while(len(y_values_second_frame) < 9):
            second_frame = next(results, None)
            y_values_second_frame = frame_coordinates(second_frame)
    
    first_point_diff = abs(y_values_first_frame[0] - y_values_second_frame[0])
    second_point_diff = abs(y_values_first_frame[5] - y_values_second_frame[5])
    first_point_threshold = first_point_diff + 15
    second_point_threshold = second_point_diff + 15

    print("First point threshold:", first_point_threshold)
    print("Second point threshold:", second_point_threshold)

    return first_point_threshold, second_point_threshold

# Starting to apply the functions when result frames are captured
if results:

    first_frame_threshold, second_point_threshold = get_starting_frames(results)

    for r in results:
        y_values = frame_coordinates(r)
        if len(y_values) >= 9:
            minimum = min(y_values)
            maximum = max(y_values)
            print("minimum y value is:", minimum)
            print("maximum y value is:", maximum)
            check_falling(y_values)
        else:
            print("No human detected.")