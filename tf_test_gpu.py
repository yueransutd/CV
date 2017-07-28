import tensorflow as tf
import dataset
import numpy as np
import cv2
import os
import time
import xlwt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


sess = tf.Session()
saver = tf.train.import_meta_graph('6400with4layers.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))
graph = tf.get_default_graph()

y_pred = graph.get_tensor_by_name("y_pred:0")

test_path='ToolsSD\\testing_data'
img_size = 128

folderName = 'D:\Object Detection Hands\\ToolsSD\\testing_data'
classes = [n for n in os.listdir(folderName)]

num_classes = len(classes)

num_channels = 3
img_size_flat = img_size * img_size * num_channels
test_images, test_ids = dataset.read_test_set(test_path, img_size,classes)
x= graph.get_tensor_by_name("x:0")

y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((1, num_classes))

def show_time(time_in_seconds):
    hours=int(time_in_seconds/3600)
    minutes=int((time_in_seconds-3600*hours)/60)
    seconds=int(time_in_seconds-3600*hours-60*minutes)
    print("Time spent so far: %02d:%02d:%02d"%(hours,minutes,seconds))
#
#def plot_confusion_matrix(cls_pred,cls_true):
#    # This is called from print_test_accuracy() below.
#
#    # cls_pred is an array of the predicted class-number for
#    # all images in the test-set.
#
#    # Get the true classifications for the test-set.
#    #cls_true = data.test.cls
#    
#    # Get the confusion matrix using sklearn.
#    cm = confusion_matrix(y_true=cls_true,
#                          y_pred=cls_pred)
#
#    # Print the confusion matrix as text.
#    print(cm)
#
#    # Plot the confusion matrix as an image.
#    plt.matshow(cm)
#
#    # Make various adjustments to the plot.
#    plt.colorbar()
#    tick_marks = np.arange(num_classes)
#    plt.xticks(tick_marks, range(num_classes))
#    plt.yticks(tick_marks, range(num_classes))
#    plt.xlabel('Predicted')
#    plt.ylabel('True')
#
#    # Ensure the plot is shown correctly with multiple plots
#    # in a single Notebook cell.
#    plt.show()
#    





file = True
if file:

    true_all_sum=0
    folder = "D:\\Object Detection Hands\\ToolsSD\\testing_data"
    files = [n for n in os.listdir(folder)]
    start=time.time()
    
    #print(files)
    #['blueScissors', 'greenScrewdriver', 'largeWireCutter', 'redScissors', 'redScrewdriver', 'smallWireCutter']
    
    write_to_xl=True
    if write_to_xl:
        workbook=xlwt.Workbook(encoding="utf-8")
        sheet1=workbook.add_sheet("Sheet1")
        #sheet1=workbook.sheet_by_index(0)
        sheet1.write(0,0,time.strftime("%d/%m/%Y"))
        sheet1.write(0,1,time.strftime("%H:%M:%S"))
    
    m=0
    
    tot=0
    for elt in files:
        #elt: class name
        print('Processing ',elt)
        matrix_part=np.zeros(6)
        img_paths=[m for m in os.listdir(folder+"\\"+elt)] 
        tot+=len(img_paths)
        #['blueScissors.10110.jpg', 'blueScissors.10181.jpg']
        #print(len(img_paths))
        i=0
        true_sum=0
        wrong_cases=[]
        for i in range(0,len(img_paths)):
            #i: same number as the number of imgs in each class
            img=cv2.imread(folder+'\\'+elt+'\\'+img_paths[i])
            #img=cv2.imread('D:\\Object Detection Hands\\ToolsSD\\testing_data\\blueScissors\\blueScissors.10110.jpg')
            img = cv2.resize(img, (img_size, img_size), cv2.INTER_LINEAR)
            x_batch = img.reshape(1, img_size_flat)

            feed_dict_testing = {x: x_batch, y_true: y_test_images}
            table = sess.run(y_pred, feed_dict=feed_dict_testing)
            
            for elt1 in table:
               
                #print(elt1) [ 0.  0.  0.  1.  0.  0.]
                
                matrix_part+=elt1
                indice = np.argmax(elt1)
                if classes[indice]==elt:
                    true_sum+=1
                    true_all_sum+=1
            
#            if len(wrong_cases)<5:
#                cv2.putText(img,classes[indice]+str(elt1[indice])+" %",(10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255) 
#                wrong_cases.append(img)
                #print(img_paths)
        #print(len(wrong_cases))         
#        for im in wrong_cases:
#                    
#            cv2.imshow("img", img)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
                
                #cv2.putText(img,classes[indice]+str(elt1[indice])+" %",(10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
        #cv2.putText(img,classes[indice]+str(elt1[indice])+" %",(10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)  
        test_acc=round(float(true_sum)/len(img_paths),4)
        total_acc=round(float(true_all_sum)/tot,4)
        print(matrix_part.round(decimals=0))
        
        sheet1.col(m).width=5000
        sheet1.write(1,m,elt)
        sheet1.write(2,m,str(test_acc*100)+'%')
        m+=1
            
        print('test_accuracy',str(round((test_acc*100),2))+'%')
        show_time(time.time()-start)
    
    #plot_confusion_matrix()    
    print('total_accuracy',str(round((total_acc*100),2))+'%')
    #workbook.save("test_result%s.xls"%(time.strftime("%d%m%Y")))
    #print("Excel created")        
        

    
        
'''img = cv2.imread(folder+"\\"+elt)
img = cv2.resize(img, (img_size, img_size), cv2.INTER_LINEAR)

x_batch = img.reshape(1, img_size_flat)

feed_dict_testing = {x: x_batch, y_true: y_test_images}
table = sess.run(y_pred, feed_dict=feed_dict_testing)

liste = []
for elt in table:
    indice = np.argmax(elt)
    cv2.putText(img,classes[indice]+str(elt[indice])+" %",(10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)'''

        #cv2.imshow("img", img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

video = not file

if video:
    videoPath = 'D:\Object Detection Hands\\videos\\test.mp4'

    cap = cv2.VideoCapture(videoPath)
    fgbg = cv2.createBackgroundSubtractorMOG2()

    i = 0
    frame_counter = 0
    countingList = np.zeros((1, len(classes)))
    while (1):
        i += 1
        ret, frame = cap.read()
        originalFrame = np.copy(frame)
        if frame is None:
            break

        frame_counter += 1
        # If the last frame is reached, reset the capture and the frame_counter
        if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            frame_counter = 0  # Or whatever as long as it is the same as next line
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)



        img = frame[0:719, 0:719, :]
        #img = cv2.Canny(img, 100, 200)
        #fullImg = np.zeros((485, 700, 3))
        #fullImg[:,:,0] = img
        #fullImg[:, :, 1] = img
        fullImg = img

        img = cv2.resize(img, (img_size, img_size), cv2.INTER_LINEAR)

        # Checking if transition between tools.
            # if meanImg <1 then it is a transition
        meanImg = np.mean(img)
        if meanImg < 1:
            countingList = np.zeros((1, len(classes)))

        x_batch = img.reshape(1, img_size_flat)

        feed_dict_testing = {x: x_batch, y_true: y_test_images}
        table = sess.run(y_pred, feed_dict=feed_dict_testing)

        originalFrame = frame


        liste = []
        for elt in table:
            indice = np.argmax(elt)
            cv2.putText(frame, classes[indice] + str(elt[indice]*100) + " %", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
            cv2.putText(frame, str(elt*100) + " %", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        255)
            cv2.putText(frame, str(classes) + " %", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        255)
            #cv2.putText(frame, str(meanImg), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       # 255)

            countingList += elt
            cv2.putText(frame, str(countingList), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        255)

        fullFrame = np.zeros((720, 300,3))
        #print("frame",np.shape(frame), np.shape(fullFrame))
        fullFrame = np.hstack((frame, fullFrame))


        # Counting the guesses per elt of list
        for elt in countingList:
            indice1 = np.argmax(elt)

            indicesSorted = np.argsort(elt)[::-1]
            if elt[indicesSorted[0]]>25 and elt[indicesSorted[2]]*1.2<elt[indicesSorted[0]] :
                realObject = classes[indice1]
                cv2.putText(frame, str(realObject), (10, 150), cv2.FONT_HERSHEY_TRIPLEX, 3,
                            150)

        usingOrb = False
        if usingOrb:
            img1 = originalFrame
            cv2.imshow('img1', originalFrame)
            img2 =cv2.imread('D:\Object Detection Hands\Tools3\\training_data\greenScrewdriver\\greenScrewdriver.1384.jpg',0) # trainImage
            # Initiate ORB detector
            orb = cv2.ORB_create()

            # find the keypoints and descriptors with ORB
            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)

            # create BFMatcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            # Match descriptors.
            matches = bf.match(des1, des2)

            # Sort them in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)
            lenMatches = len(matches)

            # Draw first 10 matches.
            cv2.putText(frame, str(lenMatches)+"ScrewDriver", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        255)

            img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
            cv2.imshow('Img3', img3)

        cv2.imshow('frame', frame)
        cv2.imshow("img", img)


        #cv2.waitKey(100)
        if cv2.waitKey(1)==27:
            break

    cap.release()