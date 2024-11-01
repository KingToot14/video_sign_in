import time
import sys
import getopt

import cv2
from recognition_model import RecognitionModel

import logging

class VideoFaceDetector:
    def __init__(self, **kwargs) -> None:
        self.recents = {}
        self.sign_in_cooldown = self.get(kwargs, 'cooldown', 300)
        self.scanning_frames = self.get(kwargs, 'scanning_frames', 1)
        
        # configure logging
        self.logger = logging.getLogger(__name__)
        
        logging.basicConfig(filename="sign_ins.log", format="%(asctime)s | %(message)s",
                            datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    
    def get(self, dict, key, default):
        if key in dict:
            return dict[key]
        return default
    
    def start_capture(self, model: RecognitionModel) -> None:
        cap = cv2.VideoCapture(0)
        
        key = None
        
        scanning_frames = self.scanning_frames
        matches = []
        
        while key != ord('q'):
            res, frame = cap.read()
            
            if res == False:
                break
            
            scanning_frames -= 1
            
            # find faces
            if scanning_frames <= 0:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                matches = model.recognize_face(rgb_frame)
                
                scanning_frames = self.scanning_frames
            
            for match in matches:
                # draw a frame around the face
                y1, x2, y2, x1 = match[0]
                y1 *= 4
                y2 *= 4
                x1 *= 4
                x2 *= 4
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                frame = cv2.rectangle(frame, (x1 - 1, y2), (x2 + 1, y2 + 20), (0, 0, 255), -1)
                name = " ".join(s.capitalize() for s in match[1].split("-"))
                frame = cv2.putText(frame, name, (x1 + 3, y2 + 15), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                
                # log the sign in (if not on cooldown)
                if name in self.recents:
                    if time.time() - self.recents[name] < self.sign_in_cooldown:
                        continue
                
                self.recents[name] = time.time()
                self.logger.info(name + " signed in")
            
            cv2.imshow("Capture", frame)
            
            key = cv2.waitKey(1)
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # parse args
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'dc:f:r:', ['debug', 'cooldown=', 'frames=', 'recdir='])
    except getopt.GetoptError:
        print('video_sign_in.py -d -c <cooldownseconds> -f <frames> -r <recognizeddir>')
        sys.exit(2)
    
    frames = 1
    cooldown = 300
    recognized_dir = "recognized_faces"
    debug = False
    
    for opt, arg in opts:
        if opt in ('-d', '--debug'):
            debug = True
        elif opt in ('-c', '--cooldown'):
            cooldown = int(arg)
        elif opt in ('-f', '--frames'):
            frames = int(arg)
        elif opt in ('r', '--recdir'):
            recognized_dir = arg
    
    video = VideoFaceDetector(scanning_frames=frames, cooldown=cooldown)
    
    # create the recognition model
    model = RecognitionModel(benchmark=debug, dir=recognized_dir)
    model.train_model()
    
    video.start_capture(model)
