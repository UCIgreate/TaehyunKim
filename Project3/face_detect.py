import cv2 as cv
import face_recognition
import os

# size of the image: 48*48 pixels
picture_size = 48

# input path for the images
folder_path = "./train"

expression = 'angry'

for i in range(0, 958):
    img = face_recognition.load_image_file(folder_path+ '/' +expression+"/"+
                  os.listdir(folder_path+ '/' + expression)[i])
    save_img = './mod_train/' + expression + '/' + str(i) +'.png'
    face = face_recognition.face_locations(img)
    if len(face)>= 1:
        cv.imwrite(save_img, img)

expression = 'disgusted'

for i in range(0, 111):
    img = face_recognition.load_image_file(folder_path+ '/' +expression+"/"+
                  os.listdir(folder_path+ '/' + expression)[i])
    save_img = './mod_train/' + expression + '/' + str(i) +'.png'
    face = face_recognition.face_locations(img)
    if len(face)>= 1:
        cv.imwrite(save_img, img)

expression = 'fearful'

for i in range(0, 1024):
    img = face_recognition.load_image_file(folder_path+ '/' +expression+"/"+
                  os.listdir(folder_path+ '/' + expression)[i])
    save_img = './mod_train/' + expression + '/' + str(i) +'.png'
    face = face_recognition.face_locations(img)
    if len(face)>= 1:
        cv.imwrite(save_img, img)

expression = 'happy'

for i in range(0, 1774):
    img = face_recognition.load_image_file(folder_path+ '/' +expression+"/"+
                  os.listdir(folder_path+ '/' + expression)[i])
    save_img = './mod_train/' + expression + '/' + str(i) +'.png'
    face = face_recognition.face_locations(img)
    if len(face)>= 1:
        cv.imwrite(save_img, img)

expression = 'neutral'

for i in range(0, 1233):
    img = face_recognition.load_image_file(folder_path+ '/' +expression+"/"+
                  os.listdir(folder_path+ '/' + expression)[i])
    save_img = './mod_train/' + expression + '/' + str(i) +'.png'
    face = face_recognition.face_locations(img)
    if len(face)>= 1:
        cv.imwrite(save_img, img)

expression = 'sad'

for i in range(0, 1247):
    img = face_recognition.load_image_file(folder_path+ '/' +expression+"/"+
                  os.listdir(folder_path+ '/' + expression)[i])
    save_img = './mod_train/' + expression + '/' + str(i) +'.png'
    face = face_recognition.face_locations(img)
    if len(face)>= 1:
        cv.imwrite(save_img, img)

expression = 'surprised'

for i in range(0, 831):
    img = face_recognition.load_image_file(folder_path+ '/' +expression+"/"+
                  os.listdir(folder_path+ '/' + expression)[i])
    save_img = './mod_train/' + expression + '/' + str(i) +'.png'
    face = face_recognition.face_locations(img)
    if len(face)>= 1:
        cv.imwrite(save_img, img)