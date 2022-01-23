#import libraries
import os
import cv2
import h5py
import numpy as np
from tkinter import *
from PIL import ImageGrab
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
#load model
from keras.models import load_model

# this is where we load the model we previously trained, MAKE SURE TO CHANGE THE PATH TO MATCH YOUR OWN LOCAL PATH
model = load_model('C:/Users/benja/documents/cs50finalproject/cs50finalproject/mnistmodelmk2.h5')


print("Model load successfully, go for APP")




#Clear canvas
def clear_widget():
    global cv
    cv.delete("all")
    
    
#Listen for mouse button motion
def activate_event(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y
 
#Actually draw the lines on the canvas
def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
   
    #do the canvas drawings
    cv.create_line((lastx, lasty, x, y), width=8, fill='black', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y


def Recognize_Digit():
    global image_number
    #image_number = 0
    filename = f'image_{image_number}.png'
    widget=cv

    #get the widget coordinates
    x=root.winfo_rootx()+widget.winfo_x()
    y=root.winfo_rooty()+widget.winfo_y()
    x1=x+widget.winfo_width()
    y1=y+widget.winfo_height()

    ImageGrab.grab().crop((x,y,x1,y1,)).save(filename)
    
    

    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    print(th)
    contours= cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]


    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        
        cv2.rectangle(image, (int(x),int(y)), (int(x)+int(w), int(y)+int(h)), (255,0,0), 1)
        top = int(0.05 * th.shape[0])
        bottom = top
        left = int(0.05 * th.shape[1])
        right = left
        th_up = cv2.copyMakeBorder(th, top, bottom, left, right, cv2.BORDER_REPLICATE)
        roi = th[y:y-h, x:x+w]
        print(roi)
        img = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)
        img = img.reshape(1,64,64,1)
        img = img / 255.0
        pred = model.predict([img])[0]
        final_pred = np.argmax(pred)
        data = str(final_pred) +' '+ str(int(max(pred)*100)) + '%'
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(image, data, (x,y-5), font, fontScale, color, thickness)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        
#create a main window first (named as root)
root = Tk()
root.resizable(0,0)
root.title("Handwritten Digit Recognition GUI App")

#Initialize few variables
lastx, lasty = None, None
image_number = 0

#create canvas for drawing
cv = Canvas(root, width=640, heigh=480, bg='white')
cv.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)

cv.bind('<Button-1>', activate_event)

#Add Buttons and Labels
btn_save = Button(text="Recognize Digit", command=Recognize_Digit)
btn_save.grid(row=2, column=0, pady=1, padx=1)
button_clear = Button(text = "Clear Widget", command = clear_widget)
button_clear.grid(row=2, column=1, pady=1, padx=1)

#mainloop() is used when your application is ready to run
root.mainloop()
