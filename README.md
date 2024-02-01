# VisiAge Fall Detection System

This project is a Fall Detection System that utilizes computer vision techniques to detect falls in a video feed. The system integrates with Azure Blob Storage to store relevant information about the detected falls.

## Features

- **Real-time Fall Detection:** Utilizes the YOLO (You Only Look Once) model to analyze video frames and identify key points of a person, enabling real-time fall detection.
- **Azure Blob Storage Integration:** Stores information about detected falls, including timestamp, video blob name, and incident status, in Azure Blob Storage.
- **API Call to App Service:** Triggers an API call to an app service endpoint when a fall is detected, providing relevant information for further alerting and analysis.

## Setup

1. **Install Dependencies:**
   - Ensure you have Python installed.
   - Install required Python packages using `pip install -r requirements.txt`.

2. **Azure Blob Storage Configuration:**
   - Set up an Azure Storage account and create a container for storing fall information and video clips.
   - Update the `AZURE_STORAGE_CONNECTION_STRING` and `CONTAINER_NAME` variables in the code with your Azure Storage account details.

3. **Run the Code:**
   - Execute the code file `fall_detection.py` to start the fall detection system.
   - The system will process the video feed, detect falls, and upload relevant information to Azure Blob Storage.

## Configuration

- **Yolo Model:**
  - The YOLO model file (`yolov8n-pose.pt`) is expected to be in the `yolo models/` directory.
  - You can replace the model file or adjust its location as needed.

- **Thresholds and Parameters:**
  - Adjust falling thresholds, time thresholds, and other parameters as needed for your specific use case. These are defined in the code under the "DEFINING VARIABLES AND CONSTANTS FOR FALLING/LAYING DOWN" section.

## Important Notes

- This project uses the FastAPI framework for handling HTTP requests. Ensure that FastAPI is suitable for your deployment environment.

- Make sure to customize the API endpoint (`url` variable in the `send_api_call` function) to match the endpoint of your app service.

## License

YOLOv8 which is used in this system is the latest version of YOLO by Ultralytics. According to the Ultralytics Licensing page, YOLOv8 repositories like YOLOv3, YOLOv5, or YOLOv8 come with an AGPL-3.0 License for all users by default. If you aim to integrate Ultralytics software and AI models into commercial goods and services without adhering to the open-source requirements of AGPL-3.0, then their Enterprise License is what you’re looking for. See the page for more details about [Ultralytics legal terms of use](https://www.ultralytics.com/legal/terms-of-use).

## Acknowledgments

- This project utilizes the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) model for pose estimation.
