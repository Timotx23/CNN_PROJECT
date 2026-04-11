from feed_data import Camera
import model.CNN_model 

def testing_model():
    dropout_prob=0.2
    camera = Camera(dropout_prob)
    camera.get_video()
testing_model()



