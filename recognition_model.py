import face_recognition
import cv2

import os
import time

import math
import numpy as np

class RecognitionModel:
    def __init__(self, **kwargs) -> None:
        self.training_dir = self.get(kwargs, 'dir', "recognized_faces")
        self.match_threshold = self.get(kwargs, 'threshold', 0.6)
        self.image_scale = self.get(kwargs, 'scale', 1.0)
        self.benchmark = self.get(kwargs, 'benchmark', False)
        
        self.encodings = []
        self.names = []
    
    # utility for getting a dictionary key
    def get(self, dict, key, default):
        if key in dict:
            return dict[key]
        return default
    
    # train a new model using the training directory specified in __init__
    def train_model(self) -> None:
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
            image_dir = os.path.join(self.training_dir, person)
            images = []
            
            for image_path in os.listdir(image_dir):
                image = cv2.imread(os.path.join(image_dir, image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
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
        face_range = (1.0 - self.match_threshold)
        linear_val = (1.0 - distance) / (face_range * 2.0)
        
        if distance > self.match_threshold:
            return f"{round(linear_val * 100, 2)}%"
        else:
            value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
            return f"{round(value, 2)}%"
    
    def get_face_index(self, index):
        i = 0
        face = 0
        prev = self.names[0]
        
        while i < index and i < len(self.names):
            if self.names[i] != prev:
                prev = self.names[i]
                face += 1
            
            i += 1
        
        return face
    
    # attempt to find any matching faces given an image
    def recognize_face(self, image):
        faces = face_recognition.face_locations(image)
        ret = []
        
        if len(faces) == 0:
            return []
        
        # test each face found
        if self.benchmark:
            start_encode = time.time()
            print("Encoding Faces")
        
        small_image = cv2.resize(image, (0, 0), fx=self.image_scale, fy=self.image_scale)
        encodings = face_recognition.face_encodings(small_image, known_face_locations=faces)
        
        if self.benchmark:
            start_encode = (time.time() - start_encode) * 1000
            print(f"- Completed in {round(start_encode)}ms")
        
        if len(encodings) == 0:
            print("Unknown person detected")
            return []
        
        face = 0
        for encoding in encodings:
            if self.benchmark:
                start_test = time.time()
                print(f"Testing face {face}")
            # test each known encoding to see if there's a match
            matches = face_recognition.compare_faces(self.encodings, encoding, tolerance=self.match_threshold)
            face_distances = face_recognition.face_distance(self.encodings, encoding)
            
            # find the best match
            best_match = np.argmin(face_distances)
            
            # if the best match is valid, print it
            if matches[best_match]:
                ret.append([faces[face], self.names[best_match]])
                # print(f"Found {self.names[best_match]} with {self.face_confidence(face_distances[best_match])} confidence")
            else:
                
                ret.append([faces[face], "unknown-person"])
                # print("Unknown person detected")
            
            if self.benchmark:
                start_test = (time.time() - start_test) * 1000
                print(f"- Completed in {round(start_test)}ms")
            
            face += 1
        
        return ret