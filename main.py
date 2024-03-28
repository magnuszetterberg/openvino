import cv2
import numpy as np
import subprocess
from openvino.runtime import Core

inputSource = "rtmp://localhost:1936/live/NewYork"
OutputDestination = "rtmp://localhost/app/NewYork-Detections"

# Initialize OpenVINO runtime
core = Core()
# Read the network and corresponding weights from a model file
model_xml = './intel/person-detection-0200/FP16-INT8/person-detection-0200.xml'
model_bin = './intel/person-detection-0200/FP16-INT8/person-detection-0200.bin'
model = core.read_model(model=model_xml)
compiled_model = core.compile_model(model=model)

# Get the names of the input and output layers
input_layer = next(iter(compiled_model.inputs))
output_layer = next(iter(compiled_model.outputs))

# Define preprocessing function
def preprocess_frame(frame, input_height, input_width):
    # Resize the frame to fit the model input
    frame = cv2.resize(frame, (input_width, input_height))
    # Convert the frame to blob
    frame = frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    frame = np.expand_dims(frame, axis=0)
    return frame

# Define postprocessing function
def draw_detections(frame, detections, threshold=0.7):
    # Iterate through detections
    for _, label, conf, x_min, y_min, x_max, y_max in detections[0][0]:
        if conf > threshold and label == 0:  # Check for confidence and if the label is 'person'
            xmin = int(x_min * frame.shape[1])
            ymin = int(y_min * frame.shape[0])
            xmax = int(x_max * frame.shape[1])
            ymax = int(y_max * frame.shape[0])
            
            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            # Add label and confidence
            cv2.putText(frame, f'Confidence: {conf:.2f}', (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


# Initialize video capture
cap = cv2.VideoCapture(inputSource, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Cannot open RTMP stream")
    exit(-1)

# Get video info for the RTMP stream
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Set up a pipe to FFmpeg
command = [
    'ffmpeg',
    '-y',  # overwrite output file if it exists
    '-f', 'rawvideo',  # input format
    '-vcodec', 'rawvideo',  # input codec
    '-s', f'{frame_width}x{frame_height}',  # size of one frame
    '-pix_fmt', 'bgr24',  # input pixel format
    '-r', str(fps),  # frames per second
    '-i', '-',  # input comes from a pipe
    '-c:v', 'libx264',  # output video codec
    '-pix_fmt', 'yuv420p',  # output pixel format
    '-preset', 'veryfast',
    '-tune', 'zerolatency',  # tune for zero latency
    '-b:v', '2500k',  # bitrate (you may want to adjust this according to your needs)
    '-f', 'flv',  # output format
    OutputDestination  # output location

]

process = subprocess.Popen(command, stdin=subprocess.PIPE)


# Read and process frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame capture failed, stream may have ended")
        break

    # Preprocess the frame
    input_shape = [dim for dim in input_layer.shape]  # Get the input shape
    p_frame = preprocess_frame(frame, input_shape[2], input_shape[3])

    # Perform inference
    results = compiled_model.infer_new_request({input_layer: p_frame})
    detections = results[output_layer]

    # Postprocess the output
    draw_detections(frame, detections)

    # Write the frame to the RTMP server via FFmpeg pipe
    process.stdin.write(frame.tobytes())

# Release resources
cap.release()
process.stdin.close()
process.wait()
cv2.destroyAllWindows()
