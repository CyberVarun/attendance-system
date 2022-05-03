from pathlib import Path
from collections import namedtuple
import face_recognition as fr
import numpy as np
import cv2

Encoding = namedtuple('Encoding', ['encoding', 'id'])

IMAGE_SIZE = 32

def preprocess_image(image: np.ndarray):
    """
    preprocess_image will preprocess the image (convert it to GRAY, increase the contrast and make all the vaues float)
    :param image: The image on which preprocessing is to be done
    :return image: The preprocessed version of the image
    """
    image = image.astype("uint8")  # Make the image into int type
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to Gray
    image = cv2.equalizeHist(image)  # Improve the contrast of the image
    image = image / 255  # Make all the values between 0 and 1
    
    return image

# def load_encodings(data_file: Path = Path('../Data/face-encodings.npy')):
#     return np.load(data_file, allow_pickle=True)

def generate_encodings(
        data_dir: Path = Path('../Data/Students'), 
        data_file: Path = Path('../Data/face-encodings.npy')
    ):
    images = []  # temporary storage for images
    
    face_images = data_dir.glob('*.png')

    for img in face_images:
        image = fr.load_image_file(img)

        encodings = fr.face_encodings(image)[0]
        images.append(Encoding(encoding = encodings, id = img.stem))

    images = np.array(images, dtype=object)

    np.save(data_file, images)

    return images

def unpack_array_of_tuples(array: np.ndarray):
    unpacked_values = []

    for i in range(len(array[0])):
        unpacked_values.append(
            np.array(
                list(
                    map(lambda encoding_dict: encoding_dict[i], array)
                )
            )
        )

    return tuple(unpacked_values)

class UnidentifiedFace:
    faces = []
    def __init__(self, img, encodings):
        self.encodings = encodings
        self.image = img
        self.name = 'Unidentified-' + str(len(UnidentifiedFace.faces))

        if not self.__is_in_face():
            UnidentifiedFace.faces.append(self)     

    def __is_in_face(self):
        if len(UnidentifiedFace.faces) == 0:
            return False
        
        matches = fr.compare_faces(UnidentifiedFace.get_encodings(), self.encodings)
        distance = fr.face_distance(UnidentifiedFace.get_encodings(), self.encodings)

        min_index = np.argmin(distance)
        if matches[min_index]:
            return True
        return False

    @staticmethod
    def get_encodings():
        return np.array(list(map(lambda face: face.encodings[0], UnidentifiedFace.faces)))

    @staticmethod
    def flush():
        UnidentifiedFace.faces = []

if __name__ == '__main__':
    face_detector_haarcascade = Path('../Data/haarcascade_frontalface_default.xml')
    face_detector = cv2.CascadeClassifier(str(face_detector_haarcascade))

    webcam = cv2.VideoCapture(0)

    identified_faces = set()
    UnidentifiedFace.flush() # clear the data

    encodings = generate_encodings()
    encodings, ids = unpack_array_of_tuples(encodings)

    while True:
        sucess, frame = webcam.read()

        if not sucess:
            print("[-] Failed to read")
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converting the image to grey for better face recognition
        faces = face_detector.detectMultiScale(frame_gray, 1.3, 5)  # Detect for faces in the current frame

        for x, y, w, h in faces: # for x_cor, y_cor, width, height of every face
            crop_img = frame[y: y+h, x: x+h]  # crop the frame just to the face using list slicing
                
            face = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            encoded_face = fr.face_encodings(face)
            
            if len(encoded_face) >= 1:
                # get the matches and the distance to identify the name of the person who is not wearing a mask 
                matches = fr.compare_faces(encodings, encoded_face)
                distance = fr.face_distance(encodings, encoded_face)
                idx = np.argmin(distance)

                if matches[idx]:  # if the comparision is true then add it to the identified faces
                    id = ids[idx]
                    identified_faces.add(id)
                    
                else:
                    UnidentifiedFace(crop_img, encoded_face)  # Add the unidentified face
        
        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1) # to force quit
        if key == ord('q') or key == ord('Q'):
            break
            
    webcam.release()
    cv2.destroyAllWindows()
    print(identified_faces)
    print(UnidentifiedFace.faces)
