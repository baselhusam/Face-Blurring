# Import open-cv lib
import cv2
import sys

def blur_face():

    """
    Blurs faces in the input image using the specified pre-trained Haar cascade classifier.

    Args:
        None, but expect the following from the command prompt:
        - img_path: string. Path to the input image.
        
    Returns:
        None. but save the blurred image

    Raises:
    - IOError: If the input image cannot be loaded or the face cascade classifier XML file cannot be loaded.

    """

    # Extract the values from the prompt
    img_path = sys.argv[1]

    # Default path for the model
    cascade_classifier_path = ".\\pretrained_models\\haarcascade_frontalface_default.xml"

    # Load the input image
    img = cv2.imread(img_path)

    # Check if the image loaded correctly
    if type(img) == None:
        raise IOError("Unable toe load the image")


    # Load the face detection cascade
    face_cascade = cv2.CascadeClassifier(cascade_classifier_path)

    # Check if the cascade classifier is loaded correctly
    if face_cascade.empty():
        raise IOError('Unable to load the face cascade classifier xml file')


    # Convert the input image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=3, minSize=(30,30))

    # Loop over all the faces and blur them
    for (x, y, w, h) in faces:

        # Create a region of interest (region) around the face
        region = img[y:y+h, x:x+w]

        # Apply Gaussian blur to the region
        blurred_region = cv2.GaussianBlur(region, (25,25), 75)

        # Replace the original region with the blurred region
        img[y:y+h, x:x+w] = blurred_region


    # Save the output image
    cv2.imwrite("cascade_output.jpg", img)

blur_face()


