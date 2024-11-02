import face_recognition
import cv2

import os
import time
from util import *

import math
import numpy as np


class RecognitionModel:
    def __init__(self, **kwargs) -> None:
        self.training_dir = get(kwargs, 'dir', "recognized_faces")
        self.match_threshold = get(kwargs, 'threshold', 0.6)
        self.benchmark = get(kwargs, 'benchmark', False)
        
        self.encodings = []
        self.names = []
    
    # train a new model using the training directory specified in __init__
    def train_model(self) -> None:
        '''Populate the model with the initial values stored in `recognized_faces`
        '''
        self.encodings = []
        self.names = []
        
        # load all training images
        if self.benchmark:
            start = time.time()
            print("Training model")
        
        all_people = []
        all_images = []
        # gather images
        for person in os.listdir(self.training_dir):
            if not os.path.isdir(person):
                continue
            
            image_dir = os.path.join(self.training_dir, person)
            images = []
            
            for image_path in os.listdir(image_dir):
                image = cv2.imread(os.path.join(image_dir, image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                if image.size != 0:
                    images.append(image)
            
            all_people.append(person)
            all_images.append(images)
        
        # Train each person
        # with ProcessPoolExecutor(max_workers=4) as executor:
        #     executor.map(self.add_person, all_people, all_images, [self for x in range(len(all_people))])
        
        for person, images in zip(all_people, all_images):
            self.add_person(person, images)
        
        if self.benchmark:
            start = (time.time() - start) * 1000
            print(f"- Completed Training in {round(start)}ms")
    
    # adds a new person to the model
    def add_person(self, name: str, images) -> None:
        '''Adds a new person to model
        
        Args:
            name (string): the name of the new person
            images (ndarray[]): The pixel values of each training image
        '''
        name = name.replace(" ", "-").lower()
        
        if self.benchmark:
            start_person = time.time()
            print(f"    Training '{name}'")
        
        image_num = 1
        for image in images:
            if self.benchmark:
                start_face = time.time()
                print(f"        Finding faces")
            faces = face_recognition.face_locations(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            if self.benchmark:
                start_face = (time.time() - start_face) * 1000
                print(f"        - Completed in {round(start_face)}ms")
            
            if self.benchmark:
                start_encoding = time.time()
                print(f"        Encoding faces")
            
            # only accept test image if 1 face is found
            if len(faces) == 0:
                print(f"Error: image {image_num} does not contain any faces.")
            elif len(faces) == 1:
                encoding = face_recognition.face_encodings(image, known_face_locations=faces)[0]
                
                # add to model
                self.encodings.append(encoding)
                self.names.append(name)
            else:
                print(f"Error: image {image_num} contains more than one face.")
            
            if self.benchmark:
                start_encoding = (time.time() - start_encoding) * 1000
                print(f"        - Completed in {round(start_encoding)}ms")
            
            image_num += 1
        
        if self.benchmark:
            start_person = (time.time() - start_person) * 1000
            print(f"    - Completed '{name}' in {round(start_person)}ms")
    
    # determine how confident we are that a face is a match
    def face_confidence(self, distance: float) -> str:
        '''Returns, as a percentage, the model's confidence in a given face
        
        Args:
            distance (float): the distance between the target face and the closest match
        
        Returns:
            string: the confidence as a percent
        '''
        face_range = (1.0 - self.match_threshold)
        linear_val = (1.0 - distance) / (face_range * 2.0)
        
        if distance > self.match_threshold:
            return f"{round(linear_val * 100, 2)}%"
        else:
            value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
            return f"{round(value, 2)}%"
    
    # attempt to find any matching faces given an image
    def recognize_face(self, image):
        '''Given an image, attempt to locate any faces, and whether or not we recognize them
        
        Args:
            image (ndarray): the target image
        '''
        faces = face_recognition.face_locations(image)
        ret = []
        
        if len(faces) == 0:
            return []
        
        # calculate the encodings of each face found in `image`
        if self.benchmark:
            start_encode = time.time()
            print("Encoding Faces")
        
        encodings = face_recognition.face_encodings(image, known_face_locations=faces)
        
        if self.benchmark:
            start_encode = (time.time() - start_encode) * 1000
            print(f"- Completed in {round(start_encode)}ms")
        
        if len(encodings) == 0:
            print("Unknown person detected")
            return []
        
        face = 0
        # attempt to match each encoding to its best match
        for encoding in encodings:
            if self.benchmark:
                start_test = time.time()
                print(f"Testing face {face}")
            # test each known encoding to see if there's a match
            matches = face_recognition.compare_faces(self.encodings, encoding, tolerance=self.match_threshold)
            face_distances = face_recognition.face_distance(self.encodings, encoding)
            
            # find the best match
            best_match = np.argmin(face_distances)
            
            if matches[best_match]:
                # if the best match is valid, print it
                ret.append([faces[face], self.names[best_match]])
            else:
                # else, return an unknown person
                ret.append([faces[face], "unknown-person"])
            
            if self.benchmark:
                start_test = (time.time() - start_test) * 1000
                print(f"- Completed in {round(start_test)}ms")
            
            face += 1
        
        return ret