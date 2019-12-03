import cv2,os
import numpy as np
from PIL import Image
from utils import import_file, printToFile, constants

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

def getImagesAndLabels(path, map_path):
    #get the path of all the files in the folder
    samples_folders=[os.path.join(path,f) for f in os.listdir(path)]
    printToFile("", map_path, False)

    #create empth face list
    faceSamples=[]
    #create empty ID list
    Ids=[]
    print(samples_folders) #################
    
    for i, sample_folder in enumerate(samples_folders):
        if not sample_folder or len(sample_folder.split('.')) > 1:
            continue

        folder = sample_folder.split("\\")[1]
        imagePaths=[os.path.join(sample_folder,f) for f in os.listdir(sample_folder)]
        print(folder) #################
        
        #getting the Id from the image
        Id=i+1

        #now looping through all the image paths and loading the Ids and the images
        for imagePath in imagePaths:

            #loading the image and converting it to gray scale
            pilImage=Image.open(imagePath).convert('L')
            #Now we are converting the PIL image into numpy array
            imageNp=np.array(pilImage,'uint8')
            # extract the face from the training image sample
            faces=detector.detectMultiScale(imageNp)
            #If a face is there then append that in the list as well as Id of it
            for (x,y,w,h) in faces:
                faceSamples.append(imageNp[y:y+h,x:x+w])
                Ids.append(Id)
        
        printToFile(str(Id) + "," + folder + "\n", map_path)

    return faceSamples,Ids


faces,Ids = getImagesAndLabels(constants['samples_folder'], constants['samples_map_file'])
recognizer.train(faces, np.array(Ids))
recognizer.write(constants['trained_folder'])

# faces,Ids = getImagesAndLabels('dataSet')
# recognizer.train(faces, np.array(Ids))
# recognizer.write('trained/trainner.yml')