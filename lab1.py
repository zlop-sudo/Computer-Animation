import pygame as pg
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np

# IMPORT OBJECT LOADER
from objloader import OBJ
import sys

import math
import random
random.seed(5)
import _thread

class keyFrame:
    # keys
    keys = np.array(((0,0,0,0,0,0),(2,1,1,20,30,30),(1,3,-1,30,40, 0),(1,-2,-1,10,0,-10),(2,-1,1,-10,-10,10), (0, 1, 2, 30, -30, 0)), dtype = float)
    keyNumber = keys.shape[0]

    a = 1 / 2.0
    M1 = np.array(((- a, 2 - a, a - 2, a), (2 * a, a - 3, 3 - 2 * a, - a), (-a, 0, a, 0), (0, 1, 0, 0)), dtype = float)
    M2 = np.array(((- 1 / 6.0, 1 / 2.0, - 1 / 2.0, 1 / 6.0), (1 / 2.0, - 1, 1 / 2.0, 0), (- 1 / 2.0, 0, 1 / 2.0, 0), (1 / 6.0, 2 / 3.0, 1 / 6.0, 0)), dtype = float)

    def __init__(self, rotationType, splineType, filename):
        # mode
        # 0 applies Fixed Angles, 1 applies Quaternions
        # 0 applies Catmul-Rom, 1 applies B-splines
        self.rotationType = rotationType
        self.splineType = splineType
        self.currentMove = np.array((0,0,0,0,0,0), dtype = float)
        self.filename = filename

        # read data
        self.vertices = None
        self.verticeNumber = 0
        self.obj = None

    # Fixed Angles rotation
    def updatePositionsFA(self):
        trans = self.currentMove[0:3]
        Rz = np.zeros([4,4], dtype = float)
        Ry = np.zeros([4,4], dtype = float)
        Rx = np.zeros([4,4], dtype = float)
        Rz[0][0] = math.cos(math.radians(self.currentMove[5]))
        Rz[0][1] = - math.sin(math.radians(self.currentMove[5]))
        Rz[1][0] = math.sin(math.radians(self.currentMove[5]))
        Rz[1][1] = math.cos(math.radians(self.currentMove[5]))
        Rz[2][2] = 1
        Rz[3][3] = 1
        Ry[0][0] = math.cos(math.radians(self.currentMove[4]))
        Ry[0][2] = math.sin(math.radians(self.currentMove[4]))
        Ry[1][1] = 1
        Ry[2][0] = - math.sin(math.radians(self.currentMove[4]))
        Ry[2][2] = math.cos(math.radians(self.currentMove[4]))
        Ry[3][3] = 1
        Rx[0][0] = 1
        Rx[1][1] = math.cos(math.radians(self.currentMove[3]))
        Rx[1][2] = - math.sin(math.radians(self.currentMove[3]))
        Rx[2][1] = math.sin(math.radians(self.currentMove[3]))
        Rx[2][2] = math.cos(math.radians(self.currentMove[3]))
        Rx[3][3] = 1
        RT = Rz.dot(Ry).dot(Rx)
        RT[0][3] = trans[0]
        RT[1][3] = trans[1]
        RT[2][3] = trans[2]
        for i in range(self.verticeNumber):
            v = self.vertices[i]
            pre = np.array([v[0], v[1], v[2], 1]).T
            cur = RT.dot(pre)
            self.obj.vertices[i] = [cur[0]/cur[3], cur[1]/cur[3], cur[2]/cur[3]]

    # Quaternions rotation
    def updatePositionsQ(self):
        trans = self.currentMove[0:3]
        Qx = self.getQuaternion(self.currentMove[3], [1, 0, 0])
        Qy = self.getQuaternion(self.currentMove[3], [0, 1, 0])
        Qz = self.getQuaternion(self.currentMove[3], [0, 0, 1])
        Q = self.quaternionMult(self.quaternionMult(Qx, Qy), Qz)
        w = Q[0]
        x = Q[1]
        y = Q[2]
        z = Q[3]
        RT = np.zeros((4,4), dtype = float)
        RT[0][0] = 1 - 2 * y * y - 2 * z * z
        RT[0][1] = 2 * x * y - 2 * w * z
        RT[0][2] = 2 * x * z + 2 * w * y
        RT[1][0] = 2 * x * y + 2 * w * z
        RT[1][1] = 1 - 2 * x * x - 2 * z * z 
        RT[1][2] = 2 * y * z - 2 * w * x
        RT[2][0] = 2 * x * z - 2 * w * y
        RT[2][1] = 2 * y * z + 2 * w * x 
        RT[2][2] = 1 - 2 * x * x - 2 * y * y
        RT[0][3] = trans[0]
        RT[1][3] = trans[1]
        RT[2][3] = trans[2]
        RT[3][3] = 1
        for i in range(self.verticeNumber):
            v = self.vertices[i]
            pre = np.array([v[0], v[1], v[2], 1]).T
            cur = RT.dot(pre)
            self.obj.vertices[i] = [cur[0]/cur[3], cur[1]/cur[3], cur[2]/cur[3]]

    def getQuaternion(self, angle, vector):
        w = math.cos(math.radians(angle / 2.0))
        sin = math.sin(math.radians(angle / 2.0))
        x = vector[0] * sin
        y = vector[1] * sin
        z = vector[2] * sin
        return [w, x, y, z]

    def quaternionMult(self, q1, q2):
        w1 = q1[0]
        v1 = np.array(q1)[1:4]
        w2 = q2[0]
        v2 = np.array(q2)[1:4]
        q_ = np.zeros((4,1), dtype = float)
        q_[0] = w1 * w2 - v1.reshape(1,3).dot(v2.reshape(3,1))
        q_[1:4] = (v2.reshape(1,3) * w1 + v1.reshape(1,3) * w2 + np.cross(v1.reshape(1,3), v2.reshape(1,3))).reshape(3,1)
        return q_

    def drawOneFrame(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        self.obj.generate()
        glTranslate(0, 0, -10)
        glCallList(self.obj.gl_list)
        pg.display.flip()
        pg.time.wait(20)

    def draw(self):
        pg.init()
        viewport = (800,600)
        hx = viewport[0]/2
        hy = viewport[1]/2
        pg.display.set_mode(viewport, OPENGL | DOUBLEBUF)

        glLightfv(GL_LIGHT0, GL_POSITION,  (-40, 200, 100, 0.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHTING)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)

        self.obj = OBJ(self.filename, swapyz=True)
        self.vertices = np.array(self.obj.vertices)
        self.verticeNumber = len(self.vertices)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        width, height = viewport
        gluPerspective(90.0, width/float(height), 1, 100.0)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_MODELVIEW)

        while True:
            # for every fragment
            for key in range(keyFrame.keyNumber - 3):
                # compute M*G
                MGs = np.empty([6, 4, 1], dtype = float)
                for i in range(6):
                    if self.splineType == 0:
                        MGs[i] = keyFrame.M1.dot(keyFrame.keys[:,i][key: key + 4].T).reshape(4, 1)
                    else:
                        MGs[i] = keyFrame.M2.dot(keyFrame.keys[:,i][key: key + 4].T).reshape(4, 1)

                # for every frame
                for dt in range(30):
                    dt = dt / 30.0
                    T = np.array((dt ** 3, dt * dt, dt, 1), dtype = float)
                    for i in range(6):
                        self.currentMove[i] = T.dot(MGs[i])
                    # update positions
                    if self.rotationType == 0:
                        self.updatePositionsFA()
                    else:
                        self.updatePositionsQ()
                    # draw
                    self.drawOneFrame()

if __name__ == "__main__":
    # mode
        # first param 0 applies Fixed Angles, 1 applies Quaternions
        # second param 0 applies Catmul-Rom, 1 applies B-splines
    KF = keyFrame(1, , 'f-16.obj')
    KF.draw()









