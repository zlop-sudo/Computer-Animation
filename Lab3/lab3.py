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

# attribute of balls
r = 64.0625 / 80
g = [0, 0, -9.80665]
restitution = 0.97

class Ball:
    def __init__(self, filename, initX, initY, initZ):
        self.obj = OBJ(filename, swapyz=True)
        self.center = []
        self.v = []
        self.init(initX, initY, initZ)

    def init(self, initX, initY, initZ):
        n = len(self.obj.vertices)
        for i in range(n):
            newV = []
            for j in range(2):
                newV.append(self.obj.vertices[i][j] / 80.0)
            newV.append((self.obj.vertices[i][2] - 64.0625) / 80.0)
            newV[0] += initX
            newV[1] += initY
            newV[2] += initZ
            self.obj.vertices[i] = newV

        self.center = [initX, initY, initZ]

        # get random init speed
        self.v = [random.random() * 10, random.random() * 10, random.random()]

    # update positions
    def move(self):
        dt = 0.05
        a = g

        # collision with wall and ground
        if self.center[2] < r:
            self.v[2] = abs(restitution * self.v[2])
        if self.center[1] > 10 - r:
            self.v[1] = - abs(restitution * self.v[1])
        if self.center[1] < - 10 + r:
            self.v[1] = abs(restitution * self.v[1])
        if self.center[0] > 10 - r:
            self.v[0] = - abs(restitution * self.v[0])
        if self.center[0] < - 10 + r:
            self.v[0] = abs(restitution * self.v[0])

        move = []
        for i in range(3):
            self.v[i] += a[i] * dt
        for i in range(3):
            move.append(self.v[i] * dt)

        for i in range(3):
            self.center[i] += move[i]

        n = len(self.obj.vertices)
        for i in range(n):
            newV = []
            for j in range(3):
                newV.append(self.obj.vertices[i][j] + move[j])
            self.obj.vertices[i] = newV

class Physics:
    def __init__(self, filename, initPositions):
        self.filename = filename
        self.initPositions = initPositions
        self.objs = []

    # compute the distance between balls
    def findCollision(self):
        n = len(self.objs)
        for i in range(n):
            center1 = np.array(self.objs[i].center)
            for j in range(i + 1, n):
                center2 = np.array(self.objs[j].center)
                if (np.sqrt(np.sum(np.square(center1 - center2)))) < r * 2:
                    self.collision(self.objs[i], self.objs[j])

    # based on momentum computing
    def collision(self, ball1, ball2):
        v1 = np.array(ball1.v)
        v2 = np.array(ball2.v)
        center1 = np.array(ball1.center)
        center2 = np.array(ball2.center)
        # direction vector
        d1_2 = center2 - center1
        d2_1 = center1 - center2
        d1_2unit = d1_2 / np.linalg.norm(d1_2)
        d2_1unit = d2_1 / np.linalg.norm(d2_1)

        v1_2 = v1.dot(d1_2unit) * d1_2unit
        v1_1 = v1 - v1_2
        v2_1 = v2.dot(d2_1unit) * d2_1unit
        v2_2 = v2 - v2_1

        if (v1_2.dot(v2_1) < 0):
            temp = v1_2
            v1_2 = v2_1
            v2_1 = temp
        elif (np.linalg.norm(v1_2) > np.linalg.norm(v2_1)):
            v1_2 = v1_2 + v2_1
            v2_1 = - v2_1
        else:
            v2_1 = v2_1 + v1_2
            v1_2 = - v1_2

        ball1.v = (v1_2 + v1_1) * restitution
        ball2.v = (v2_1 + v2_2) * restitution

        ball1.move()
        ball2.move()

    def addGround(self):
        ground_vertices = (
            (-10,10,0),
            (-10,-10,0),
            (10,-10,0),
            (10,10,0)
        )
        glBegin(GL_QUADS)
        for vertex in ground_vertices:
            glColor3fv((1,1,1.5))
            glVertex3fv(vertex)
        glEnd()

    def drawOneFrame(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        for ball in self.objs:
            ball.obj.generate()

        glTranslate(0, 5, -17)
        glRotatef(-30,1,0.1,0)
        glRotatef(-30,0,0,1)
        self.addGround()
        for ball in self.objs:
            glCallList(ball.obj.gl_list)
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

        for init in initPositions:
            self.objs.append(Ball(self.filename, init[0], init[1], init[2]))

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        width, height = viewport
        gluPerspective(90.0, width/float(height), 1, 100.0)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_MODELVIEW)

        while True:
            # for every timestamp
            for dt in range(3000):
                self.findCollision()
                # move each ball
                for ball in self.objs:
                    ball.move()
                # draw
                self.drawOneFrame()

if __name__ == "__main__":
    initPositions = [[2, 2, 5], [2, -2, 5], [-2, -2, 5], [-2, 2, 5], [3, -5, 5], [5, -5, 5]]
    physics = Physics('ball.obj', initPositions)
    physics.draw()










