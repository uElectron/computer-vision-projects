############################################################################################################################
## Author: UDHAY KUMAR MUGATI                                                                                             ##
## Version: 1.0 (Base Stable)                                                                                             ##
## Description: Searches recursively under the given directory for .jpg image files and generate appropriate .json files  ##
##               with contours detection and labelling the points for working with LabelMe program                        ##
############################################################################################################################
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import base64

#List of jpg files with directory Paths
flist = []
template = {
     'version': '4.5.6',
     'flags': {},
     'shapes': [],
     'imagePath': '',
     'imageData': '',
     'imageHeight': 0,
     'imageWidth': 0
           }

def getFilesList(dirName):
    global flist
    if dirName.endswith('/'):
        dirpath = dirName
    else:
        dirpath = dirName + '/'
    dataDir = os.listdir(dirName)
    for fname in dataDir:
        if fname.endswith('.jpg'):
            flist.append(dirpath + fname)
        elif fname.count('.') == 0:#It is a directory
            getFilesList(dirpath + fname )

def getContoursPts(imagePath):
    img = cv.imread(imagePath)
    greyimg = img[:,:,2].copy()
    gaus_grey = cv.GaussianBlur(greyimg,(7,7),10)
    gbw_gaus = cv.threshold(gaus_grey,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)[1]
    contours_gaus,heirarcy_gaus = cv.findContours(gbw_gaus, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    listaccepted = []
    for i in range(len(contours_gaus)):
        if heirarcy_gaus[0][i][3] == -1 :
            if len(contours_gaus[i]) > 100:
                listaccepted.append([i, len(contours_gaus[i])])
    try:
        fjson = open( imagePath.replace('.jpg','.json',1) , 'r')
        data = json.load(fjson)
    except:
        print('\nOpening' + imagePath[:-3] + 'json in Writing Mode')
        fjson = open( imagePath.replace('.jpg','.json',1) , 'w')
        data = template.copy()
        data['imagePath'] = imagePath.split( '/',-1)[-1]
        data['imageHeight'] , data['imageWidth'] ,null = img.shape
        data['imageData'] = base64.b64encode(cv.imencode('.jpg', img)[1]).decode()
        fjson.write(json.dumps(data))
    fjson.close()
    pts = []
    singleCnt = {
    'label': 'S',
     'points': [],
     'group_id': None,
     'shape_type': 'polygon',
     'flags': {}
     }
    for i,null in listaccepted:
        cnt = contours_gaus[i]
        epsilon = 0.01*cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)
        cntpts = approx.reshape(-1,2)
        singleCnt['points'] = cntpts.tolist()
        pts.append(singleCnt.copy())
        singleCnt['points'] = []
    fp = open( imagePath.replace('.jpg','.json',1) , 'w')
    data['shapes'] = pts[:]
    fp.write(json.dumps(data))
    fp.close()

    

    
    
    
            
            
nameofDirectory = 'data' + '/'           

getFilesList(nameofDirectory)

for fname in flist:
    getContoursPts(fname)

print('\n Operation Completed !')
