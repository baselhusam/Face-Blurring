# Import packages
from ultralytics import YOLO
import cv2
import sys

def yolo_face_blur():
    
    """
    Applies blurring to detected faces in an image using a YOLO face detection model.

    Args:
        None, but expect the following from the command prompt
            img_path (str): Path to the input image to be processed.
            
    Returns:
        None. but save the blurred image.

    """

    # Extract the values from the prompt
    img_path = sys.argv[1]

    # Default path for the model
    weights_path = ".\\pretrained_models\\YOLO_Model.pt"

    # Build the model
    model = YOLO('yolov8n.pt')
    model = YOLO(weights_path)

    # start detecting
    results = model(img_path)
    img = cv2.imread(img_path)

    # loop over detections
    for result in results:
        
        # get the bounding boxes results
        boxes = result.boxes

        if boxes:

            # loop over each box for each detect
            for box in boxes:

                # assigning the cordinate for the bounding box
                bouding_cordinate = box.xyxy
                x1, y1, x2, y2 = ( int(bouding_cordinate[0,0].item()), int(bouding_cordinate[0,1].item()), 
                                   int(bouding_cordinate[0,2].item()), int(bouding_cordinate[0,3].item()) )

                # delete all the negative cordinates
                x1, y1, x2, y2 = max(x1,0), max(y1, 0), max(x2, 0), max(y2, 0)

                # select the region & apply Gaussian Blur & put it on the original image
                region = img[y1:y2, x1:x2]
                blurred_region = cv2.GaussianBlur(region, (25,25), 100)
                img[y1:y2, x1:x2] = blurred_region


    # Save the image
    cv2.imwrite("yolo_output.jpg", img)

yolo_face_blur()
