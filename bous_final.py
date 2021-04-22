#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import os
import threading, time
import sys
import numpy as np
import cv2 as cv2# OpenCV-Python
import math
import copy
import tf
import tf2_ros
from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import Twist, Pose, TransformStamped
from tf.transformations import *
from ar_track_alvar_msgs.msg import AlvarMarkers, AlvarMarker
sys.dont_write_bytecode = True
sys.path.append( os.path.abspath(os.path.join(os.path.dirname(__file__),"../../../../common/imp")) ) # get import pass : DSR_ROBOT.py 

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

MM2M = 1.0 / 1000.0
M2MM  = 1000.0

AR_MARKER_FRAME_PREFIX_ = 'ar_marker_'
AR_TEMP_FRAME_PREFIX_  = 'ar_temp_'
AR_CALIB_FRAME_PREFIX_  = 'ar_calib_'
AR_TARGET_FRAME_PREFIX_ = 'ar_target_'
CAMERA_FRAME_PREFIX_    = 'camera_link'

OFFSET_FROM_TARGET_X  = 0.0   * MM2M # 보정 [mm]
OFFSET_FROM_TARGET_Y  = 0.0   * MM2M # 250.0 [mm]
OFFSET_FROM_TARGET_Z  = 175.0 * MM2M # [mm]
OFFSET_FROM_TARGET_RX = 180.0
OFFSET_FROM_TARGET_RY = 0.0
OFFSET_FROM_TARGET_RZ = -90.0

# for single robot 
ROBOT_ID     = "R_001/"
ROBOT_MODEL  = "dsr/"

import DR_init
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
from DSR_ROBOT import *

################

class DSRCleaningRobot():
  def __init__(self):
    rospy.init_node('Surface_Cleaning', anonymous=True)
    self.sub1 = rospy.Subscriber('/R_001/ar_pose_marker', AlvarMarkers, self.get_marker_position)

    self.listener = tf.TransformListener()
    self.position_x = 0
    self.position_y = 0
    self.position_z = 0

      #### 이 부분 맞는지 모르겠음 ####
  def get_tf_BaseCam(self):
    self.listener.waitForTransform("/base_0", "/camera_link", rospy.Time(), rospy.Duration(0.5))
    (trans,rot) = self.listener.lookupTransform("/base_0", "/camera_link", rospy.Time(0))
    return trans, rot

  def QuartToRotMat(self, rot):
    x = rot[0]
    y = rot[1]
    z = rot[2]
    w = rot[3]

    Mat = np.zeros((3,3))

    Mat[0, 0] = 1 - 2*(y*y + z*z)
    Mat[1, 0] = 2*(x*y - z*w)
    Mat[2, 0] = 2*(x*z + y*w)

    Mat[0, 1] = 2*(x*y + z*w)
    Mat[1, 1] = 1 - 2*(x*x + z*z)
    Mat[2, 1] = 2*(y*z - x*w)

    Mat[0, 2] = 2*(x*z - y*w)
    Mat[1, 2] = 2*(y*z + x*w)
    Mat[2, 2] = 1 - 2*(x*x + y*y)

    return Mat

  def get_marker_position(self, msg):

    n = len(msg.markers)
    if(n == 0):
      print("#####No AR TAGS ARE FOUND!!#####")

    else:
      idx = 0
      for x in msg.markers:
        ## Some of the frames
        ar_marker_frame = AR_MARKER_FRAME_PREFIX_ + str(msg.markers[idx].id)
        ar_temp_frame   = AR_TEMP_FRAME_PREFIX_   + str(msg.markers[idx].id)
        ar_calib_frame  = AR_CALIB_FRAME_PREFIX_  + str(msg.markers[idx].id)
        base_frame      = 'base_0'
        reference_frame = 'link6'

        ## Look up TF of "base_0" - "ar_tag" for further calculation
        self.listener.waitForTransform(base_frame, ar_marker_frame, rospy.Time(), rospy.Duration(0, 5))
        (trans_base2ref,rot_base2ref) = self.listener.lookupTransform(base_frame, ar_marker_frame, rospy.Time(0))
        if self.position_x == 0 and self.position_y == 0 and self.position_z == 0:
          self.position_x = trans_base2ref[0]
          self.position_y = trans_base2ref[1]
          self.position_z = trans_base2ref[2]
          print("#####THE AR TAG IS DETECTED!#####")
          offset = np.array([self.position_x, self.position_y, self.position_z])
          img = self.MapGeneration()
          dist = int(136)
          Movel_sep = Get3DPath(img, dist, 5, offset)
          print "Operating Cleaning Process"
          task_compliance_ctrl([500, 4500, 4000, 1000, 1000, 1000])
          rotate_on()
          for pos in Movel_sep:
            #print('move to point')
            #print(pos)
            movel(pos, v=100)
          movej([25.3, 44.69, 50, 10.3, 30.9, 0], v=20, a=20)
          rotate_off()
          release_compliance_ctrl()
          print 'Done!'



  def MapGeneration(self):
    img = Map()
    dist = int(136)
    img = Map_paddle(img, dist)
    return img


########################## Boustrophedon Cell Decomposition #################################

#rowFreeSpace(): returns the number of partition of each row(output), the list of index of starting point of a partition (sidy), andthe list of index of ending point of a partition (eidy)

def rowFreeSpace(Row):
  l = Row.shape[0] # the length of row
  sidy = []
  eidy = []
  if Row[0, 0] == 255: # when the first pixel of the row is white
    output = 1;
    sidy.append(0)
  else:
    output = 0;
  for i in range(1,l):
    if (Row[i-1, 0] == 0)  and (Row[i ,0] == 255): # the point where the partiotion starts
      output = output + 1
      sidy.append(i)
    elif (Row[i-1, 0] == 255)  and (Row[i ,0] == 0): # the point where the partiotion ends
      eidy.append(i-1)
    elif (i == l - 1) and (Row[i-1, 0] == 255): # where the row ends.
      eidy.append(i)
  return output, sidy, eidy

# FindCell returns: the number of cells in the image and row index of where partions start and end.

def FindCell(output, img):
  idx = []
  cell_no, a, b  = rowFreeSpace(img[0,:,:])
  for i in range(1, len(output)):
    if (output[i - 1] != output[i]):
      output1, sidy1, eidy1 = rowFreeSpace(img[i - 1, :, :])
      output2, sidy2, eidy2 = rowFreeSpace(img[i, :, :])

      if (output1 > output2):
        idx.append(i)
        cell_no = cell_no + output2
      elif (output1 < output2):
        idx.append(i)
        cell_no = cell_no + output2
  print()
  check1 = 0;
  for m in range(len(output)):
    if (output[m] > 0) and (check1 == 0):
      st = m
      check1 = 1
    if (output[m] > 0):
      ed = m
  if (st != idx[0]):
      idx.insert(0, st)
  if (ed != idx[len(idx) - 1]-1):
      idx.append(ed)
  return idx, cell_no


# bcdPath: returns the 2d-path of the each cell

def bcdPath(startRow, endRow, startColumn, endColumn, dist):
  flag = 0
  path = []
  pathl = []
  Rows = (endRow - startRow) + 1 # the number of row of each cell
  LengthSc = len(startColumn)
  LengthEc = len(endColumn)
  if (Rows != LengthSc):
    print("\n The num of Rows and Length of Col P of 1 not equal in Row %d",startRow)
  if (startRow > 0):
    inter = startRow - 1
    loc = []
    for i in range(inter):
      loc.append(1)
    startColumn = loc + startColumn
    endColumn = loc + endColumn
  for i in range(startRow, endRow + 1, dist):
    if (flag == 0): # left to right
      pathl.append([i, startColumn[i]]) # add staring and end point in each row to 'movel'
      pathl.append([i, endColumn[i]])
      for j in range(startColumn[i], endColumn[i]+1):
        path.append([i,j]) 
      flag = 1
    elif(flag == 1): # right to left
      pathl.append([i, endColumn[i]])
      pathl.append([i, startColumn[i]])
      for j in range(endColumn[i], startColumn[i]-1):
        path.append([i,j])
      flag = 0
  return path, pathl

# findPath: returns the 2d-path of the whole map

def findPath(img, idx, no_of_cells, sidy, eidy, dist):
  Path = []
  Movel = []
  q = []
  l = []
  m = []
  for i in range(img.shape[0]):
    l.append(len(sidy[i]))
    m.append(len(eidy[i]))
  for i in range(len(idx)-1):
    for j in range(l[idx[i]]):
      startColumn = []
      endColumn = []
      for k in range(idx[i], idx[i+1]):
        startColumn.append(sidy[idx[i]][j]) # iterate the staring column of each cell
        endColumn.append(eidy[idx[i]][j]) # iterate the ending column of each cell
      endRow = idx[i+1]-1 # iterate the ending row of each cell
      startRow = idx[i] # iterate the starting row of each cell 
      path, movel = bcdPath(startRow, endRow, startColumn, endColumn, dist) # generate the path of each cell
      Path.append(path)
      Movel.append(movel)
  return Path, Movel


	


# Get3DPath: generate 3-D Path with calibration

def Get3DPath(img, dist,num_sep, offset):
  x = [] # group of output
  sidy = [] # group of sidy
  eidy = [] # group of eidy

  for i in range(img.shape[0]):
    output, sidy_, eidy_ = rowFreeSpace(img[i,:,:])
    x.append(output)
    sidy.append(sidy_)
    eidy.append(eidy_)

  idx, no_of_cells = FindCell(x, img)
  Path, Movel = findPath(img, idx, no_of_cells, sidy, eidy, dist)
  Movel_pos = [] # xyz roll pitch yaw

  #because Movel List is divided by each cell, we concatnated it and make these 2D point into 3D points
  # transform xy point of the Movel points in each cells to [x, y, z roll, pitch, yaw] lists
  for Cell in Movel:
    cell = copy.deepcopy(Cell)
    modify = [0, 0, 200]
    length = len(cell)

    pos = copy.deepcopy(cell[0])
    pos.extend([0])
    pos = np.array(pos)
    modify = 1000*offset + np.array(modify)
    pos = pos + modify
    pos = pos.tolist()
    pos.extend([0, 180, 0])
    Movel_pos.append(pos)

    for pos in cell:
      pos = copy.deepcopy(pos)
      pos.extend([0])
      pos = np.array(pos)
      modify = [0, 0, 85]
      modify = 1000*offset + np.array(modify)
      pos = pos + modify
      pos = pos.tolist()
      pos.extend([0, 180, 0])
      Movel_pos.append(pos)
    
    pos = copy.deepcopy(cell[length-1])
    pos.extend([0])
    modify = [0, 0, 200]
    pos = np.array(pos)
    modify = 1000*offset + np.array(modify)
    pos = pos + modify
    pos = pos.tolist()
    pos.extend([0, 180, 0])
    Movel_pos.append(pos)

  Movel_pos = np.array(Movel_pos) # list to array
  Movel_sep = []
  num_move = Movel_pos.shape[0] # position 개수

  for i in range(num_move-1):
    for j in range(num_sep):
      pos = (1-j/num_sep)*Movel_pos[i,:] + j/num_sep*Movel_pos[i+1,:] # divide the movel points into 5 parts
      pos = pos.tolist() # array to list
      Movel_sep.append(pos) # divided points

  return Movel_sep

#rot
def rotate_on(pin = 12):
  if get_digital_output(pin) == 0:
    set_digital_output(pin,1)
def rotate_off(pin = 12):
  if get_digital_output(pin) == 1:
    set_digital_output(pin,0)
#############################################################################################
#ASSUMTION: MAP IS GIVEN
#MAP
def Map():
  img = np.zeros((500,500,3), np.uint8) # call 500px(mm) X 500px(mm) size map
  cv2.rectangle(img, (200,200), (300,300), (255,255,255), -1) # generating obstacle in the middle of the map
  ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
  return img
    
#MAP with obstcles paddle
def Map_paddle(img, dist=30): # MAP, diameter(mm) of the tool

  row_img = img.shape[0]
  col_img = img.shape[1]
  img_paddle = np.zeros((dist + row_img, dist + col_img, 3), np.uint8)
  radius = int(dist/2)
  img_paddle[dist : row_img , dist : col_img , :] = img[radius : row_img - radius , radius : col_img - radius , :]

  for i in range(row_img):
    for j in range(col_img):
      if img[i, j , 0] == 0:
        img_paddle[i : i + dist, j : j + dist] = 0
  img = img_paddle[radius : row_img + radius , radius : col_img + radius ]
  return img
##############################################################################################

def shutdown():
  print "shutdown time!"
  print "shutdown time!"
  print "shutdown time!"

  return 0

def thread_subscriber():
  rospy.Subscriber('/'+ROBOT_ID +ROBOT_MODEL+'/state', RobotState, msgRobotState_cb)
  rospy.spin()
  #rospy.spinner(2)
  
if __name__ == "__main__":
  cleaning = DSRCleaningRobot()
  rospy.on_shutdown(shutdown)
  # pub_stop = rospy.Publisher('/'+ROBOT_ID +ROBOT_MODEL+'/stop', RobotStop, queue_size=10)

  set_velx(30,20)  # set global task speed: 30(mm/sec), 20(deg/sec)
  set_accx(60,40)  # set global task accel: 60(mm/sec2), 40(deg/sec2)
  velx = [0, 0]
  accx = [0, 0]
  JReady = [-180.0, 71.4, -145.0, 0.0, -9.7, 0.0]
  
  init_pos = [96.4, -2.2, 112.0, -2.4, 68.5, 28.4]
  #movej(init_pos,v=20,a=20)

  
  

  while not rospy.is_shutdown():
    pass
