import cv2
import platform
from cv2_enumerate_cameras import enumerate_cameras
import torch
import numpy as np
import model.CNN_model 

HEIGHT = 32
WIDTH = 32
RGB = 3
device = model.CNN_model.to_devices()

class PreProcessCamera:
    def __init__(self):
        self.os = self.get_os()
        self.path = self.get_camera_path()
    def get_os(self):
        """
        Autodetect which OS is being used  -> important for the detection of the camera name / index.
        Different os have different backend values to access for camera 
        """
        if platform.system() == 'Darwin':
            backend = cv2.CAP_AVFOUNDATION  #mac
        elif platform.system() == 'Windows':
            backend = cv2.CAP_MSMF #windows
        else:
            backend = cv2.CAP_V4L2 #linux
        return backend
    
    def get_camera_path(self) -> int:
        """
        Finding the correct camera to start the thread in the correct index
        FaceTime HD Camera ~ any macbook face camera
        """
        cams: list = enumerate_cameras(self.os) 
        for cam in cams:
            # Test the index before returning it
            if cam.name.lower() == "facetime hd camera": # -> change this in future sothat it works with the pi
                test_cap = cv2.VideoCapture(cam.index, self.os)
                if test_cap.isOpened():
                    success, _ = test_cap.read()
                    test_cap.release()
                    
                    if success:
                        return cam.index
                    else:
                        print(f"Warning: Index {cam.index} matched name but failed to read a frame. Trying next...")
                else:
                    print(f"Warning: Index {cam.index} matched but failed to open.")

        raise ValueError(f"Camera '{cam.name}' could not be found or opened for use.")
    
    

class Camera(PreProcessCamera):
    def __init__(self, dropout_prob):
        super().__init__()
        self.load_model = LoadModel(dropout_prob)

       
    def get_video(self):
        video = cv2.VideoCapture(self.path, self.os)
        frame_counter = 0
        while True:

            success, frame = video.read()
            if success:
                frame_counter +=1
                if frame_counter % 3 ==0: # make system only work at 10 fps for now in order to not overload the cnn model
                    tensorizedframe = TensorizedFrame(frame)
                    correct_frame_format: torch.Tensor =  tensorizedframe.correct_tensor()
                    #call model with correct_frame_format
                    # model_call = self.load_model(correct_frame_format)
            else:
                raise ValueError ("Failed to verify video")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()

    def is_testing(self):
        ...
    
    





class TensorizedFrame:
    def __init__(self,frame) -> None:
        """This class has 1 major task which is to prepare the frame for the CNN"""
        self.frame: np.ndarray =frame
        self.corrected_frame: np.ndarray = self._corrected_cnn_format()
        self.final_correct_tensor_format: torch.Tensor = self._set_tensor_dimentions()
    


    def _corrected_cnn_format(self) -> np.ndarray:
        correct_frame_size: np.ndarray = cv2.resize(self.frame, (WIDTH,HEIGHT)) #my model was trained on 32 x 32 images so it is good to keep that format up 
        correct_format: np.ndarray = cv2.cvtColor(correct_frame_size, cv2.COLOR_BGR2RGB) # Tensors require RGB but cv2 outputs BGR meanuing ut must be converted
        return correct_format

    def _set_tensor_dimentions(self) -> torch.Tensor:
        tensor_frame = self.corrected_frame / 255.0
        tensor_frame: torch.Tensor = torch.tensor(tensor_frame).float()
        tensor_frame: torch.Tensor = tensor_frame.permute(2, 0, 1)
        
        tensor_frame: torch.Tensor = (tensor_frame -0.5)/0.5
        tensor_frame: torch.Tensor = tensor_frame.unsqueeze(0)
        return tensor_frame
    
    def correct_tensor(self):
        return self.final_correct_tensor_format

    

class LoadModel:
    def __init__(self):
        self.model = model.CNN_model.SimpleCNN_dropout(dropout_prob=0.2).to(self.device)
        self.model.load_state_dict()# Here i have to add the finished trained weights
        self.model.to(device)
        self.model.eval()
        

    def set_frame_to_model(self,frame):
        return self.model(frame)