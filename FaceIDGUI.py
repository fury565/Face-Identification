#osjetljiv na odbljesak naočala(manjak preciznosti za oči)
#prekrivanje oka povećava podudaranje???
#šeširi??
#brada-brada 95-100  brada-nebrada 115-125
import tkinter
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
import cv2 as cv
import mediapipe as mp
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from scipy.spatial import distance
from matplotlib import pyplot

def predict_face(model,facearray):
        pixels = facearray.astype('float32')
        samples = np.expand_dims(pixels, axis=0)
        samples = preprocess_input(samples, version=2)
        yhat = model.predict(samples,verbose=0)
        return yhat[0][0][0]
def compare_predictions(first,second):
        result=distance.euclidean(first,second)
        return result
def extract_face(detector,filename, required_size=(224, 224)):
        with detector.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detection:
                image = cv.imread(filename)
                results = face_detection.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
                if not results.detections:
                        return []
                for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x1=int(bbox.xmin*width)
                        x2=int((bbox.xmin+bbox.width)*width)
                        y1=int(bbox.ymin*height)
                        y2=int((bbox.ymin+bbox.height)*height)
                        face = image[y1:y2, x1:x2]
                        face = Image.fromarray(face)
                        face = face.resize(required_size)
                        pixels = np.asarray(face)
                        #pixels=cv.GaussianBlur(pixels,(3,3),cv.BORDER_DEFAULT)
                return pixels

def select_file():
        global selectedfile, idPrediction
        filetypes = (('jpg files', '*.jpg'),('All files', '*.*'))
        selectedfile = filedialog.askopenfilename(title='Open a file',initialdir='/',filetypes=filetypes)
        if len(selectedfile)!=0:
                facearray=extract_face(detector,selectedfile)
                if len(facearray)==0:
                        text_holder.configure(text="No face detected")
                        button1["state"]="disabled"
                else:
                        text_holder.configure(text="Face detected")
                        idPrediction=predict_face(model,facearray)
                        button1["state"]="normal"

def open_camera():
    open_button["state"]="disabled"
    button1["state"]="disabled"
    global frame, cam, idPrediction
    cam = cv.VideoCapture(0)
    width = cam.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cam.get(cv.CAP_PROP_FRAME_HEIGHT)
    with detector.FaceDetection(model_selection=1, min_detection_confidence=0.6) as face_detection:
            while True:
                ret, frame = cam.read()
                if not ret:
                    break
                image=frame.copy()
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                image.flags.writeable = True
                results = face_detection.process(image)
                if results.detections:
                        for detection in results.detections:
                                mp_drawing.draw_detection(frame, detection)
                                bbox = detection.location_data.relative_bounding_box
                                x1=int(bbox.xmin*width)
                                x2=int((bbox.xmin+bbox.width)*width)
                                y1=int(bbox.ymin*height)
                                y2=int((bbox.ymin+bbox.height)*height)
                                face = frame[y1:y2, x1:x2]
                                try:
                                        face = Image.fromarray(face)
                                        face = face.resize((224,224))
                                        pixels = np.asarray(face)
                                        prediction=predict_face(model,pixels)
                                        compare=compare_predictions(prediction,idPrediction)
                                        #cv.putText(frame,str(int(compare)),(x2,y2),cv.FONT_HERSHEY_SIMPLEX,0.75,(255,0,0))
                                        if compare>130:
                                                text_holder.configure(text="ID doesn't match",bg='#f00')
                                        elif compare>120:
                                                text_holder.configure(text="Possible match",bg='#ff0')
                                        else:
                                                text_holder.configure(text="ID matches",bg='#0f0')
                                except Exception as e:
                                        continue
                img_update = ImageTk.PhotoImage(Image.fromarray(frame))
                image_holder.configure(image=img_update)
                image_holder.image=img_update
                image_holder.update()

                k = cv.waitKey(1)
                if k%256 == 27:
                    print("Escape hit, closing...")
                    close_camera()
                    break

def close_camera():
    global cam
    cam.release()
    open_button["state"]="normal"
    button1["state"]="normal"

global cam, selectedfile,idPrediction
red=(255,0,0)
yellow=(0,255,0)
green=(0,0,255)
cam = cv.VideoCapture(0)
width = cam.get(cv.CAP_PROP_FRAME_WIDTH)
height = cam.get(cv.CAP_PROP_FRAME_HEIGHT)
cam.release()
detector=mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
model = VGGFace(model='resnet50',include_top=False)
idPrediction=[]

window=tkinter.Tk()
window.title("Face Identification")
frame=np.random.randint(0,255,[100,100,3],dtype='uint8')
img = ImageTk.PhotoImage(Image.fromarray(frame))
image_holder=tkinter.Label(window)
image_holder.grid(row=0,column=0,columnspan=3,pady=1,padx=10)
message="Waiting for ID selection"
text_holder=tkinter.Label(window,text=message,bg='#fff')
text_holder.grid(row=1,column=1,pady=1,padx=10)
buttonsize=10
button1=tkinter.Button(window,text="Start",command=open_camera,height=5,width=20)
button1.grid(row=1,column=0,pady=10,padx=10,rowspan=2)
button1.config(height=1*buttonsize,width=20)
button1["state"]="disabled"
button2=tkinter.Button(window,text="Stop",command=close_camera,height=5,width=20)
button2.grid(row=1,column=2,pady=10,padx=10,rowspan=2)
button2.config(height=1*buttonsize,width=20)
open_button = tkinter.Button(window,text='Select file',command=select_file)
open_button.grid(row=2,column=1,pady=1,padx=1)
window.mainloop()
