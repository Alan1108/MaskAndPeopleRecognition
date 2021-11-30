# import the necessary packages
import numpy as np
import cv2
from keras.models import load_model

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#Entrenar la red neuronal y luego escoger el numero de modelo dependiendo de las veces que se haya entrenado
model=load_model("D:\Repos\\MaskAndPeopleRecognition\\model2-001.model")
haarcascade = cv2.CascadeClassifier('D:\Programas\Python\Python39\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
results={0:'without mask',1:'mask'}
GR_dict={0:(0,0,255),1:(0,255,0)}

cv2.startWindowThread()

rect_size = 2
# open webcam video stream
cap = cv2.VideoCapture(0)

# the output will be written to output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))

while(True):
    # Captura cuadro por cuadro
    ret, frame = cap.read()
    # redimensionando para una mejor deteccion
    frame = cv2.resize(frame, (640, 480))
    # transformando cada captura a escala de grises para una mejor deteccion
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # deteccion de las personas
    # devuelve los rectangulos para los objetos detectados
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    rerect_size = cv2.resize(frame, (frame.shape[1] // rect_size, frame.shape[0] // rect_size))
    faces = haarcascade.detectMultiScale(rerect_size)
    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f] 
        
        face_img = frame[y:y+h, x:x+w]
        rerect_sized=cv2.resize(face_img,(150,150))
        normalized=rerect_sized/255.0
        reshaped=np.reshape(normalized,(1,150,150,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
        label=np.argmax(result,axis=1)[0]      
        cv2.rectangle(frame,(x,y),(x+w,y+h),GR_dict[label],2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),GR_dict[label],-1)
        cv2.putText(frame, results[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    for (xA, yA, xB, yB) in boxes:
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)
    
    out.write(frame.astype('uint8'))
    # Despliega el cuadro de resultado
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
cv2.waitKey(1)