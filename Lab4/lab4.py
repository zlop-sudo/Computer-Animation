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

# attribute of boids
max_force = 0.2
max_speed = 3.5
perception = 8

# coefficients of behavior flock effect
c_alignment = 1
c_cohesion = 0.6
c_separation = 1.2

class Bird:
    def __init__(self, filename, initX, initY, initZ):
        self.obj = OBJ(filename, swapyz=True)
        self.center = np.zeros(3, dtype=float)
        self.v = np.zeros(3, dtype=float)
        self.a = np.zeros(3, dtype=float)
        self.init(initX, initY, initZ)

    def init(self, initX, initY, initZ):
        n = len(self.obj.vertices)
        for i in range(n):
            newV = []
            for j in range(2):
                newV.append(self.obj.vertices[i][j] / 20.0)
            newV.append(self.obj.vertices[i][2] / 20.0)
            newV[0] += initX
            newV[1] += initY
            newV[2] += initZ
            self.obj.vertices[i] = newV

        self.center = np.array([initX, initY, initZ], dtype=float)

        # get random initial speed
        self.v = np.array([random.random() * max_speed, random.random() * max_speed, random.random() * max_speed * 0.25])

    # update positions
    def move(self):
        dt = 0.1

        self.v += self.a * dt

        # speed limit
        if np.linalg.norm(self.v) > max_speed:
            self.v = self.v / np.linalg.norm(self.v) * max_speed

        move = np.zeros(3, dtype=float)
        move += self.v * dt

        preCenter = np.copy(self.center)
        self.center += move
        self.crossEdges(self.center)
        move = self.center - preCenter

        n = len(self.obj.vertices)
        for i in range(n):
            newV = []
            for j in range(3):
                newV.append(self.obj.vertices[i][j] + move[j])
            self.obj.vertices[i] = newV

    # fly across edges and come to the other side
    def crossEdges(self, v):
        if v[2] < -10:
            v[2] = 9
        if v[2] > 10:
            v[2] = -9
        if v[1] > 20:
            v[1] = -19
        if v[1] < -20:
            v[1] = 19
        if v[0] > 20:
            v[0] = -19
        if v[0] < -20:
            v[0] = 19

    def apply_behaviour(self, boids):
        alignment = self.align(boids)
        cohesion = self.cohesion(boids)
        separation = self.separation(boids)

        self.a += alignment
        self.a += cohesion
        self.a += separation

    def align(self, boids):
        steering = np.zeros(3, dtype=float)
        count = 0
        avg_vector = np.zeros(3, dtype=float)
        for boid in boids:
            if np.linalg.norm(boid.center - self.center) < perception:
                avg_vector += boid.v
                count += 1
        if count > 0:
            avg_vector /= count
            avg_vector = (avg_vector / np.linalg.norm(avg_vector)) * max_speed
            steering = avg_vector - self.v
        return steering * c_alignment

    def cohesion(self, boids):
        steering = np.zeros(3, dtype=float)
        count = 0
        center_of_mass = np.zeros(3, dtype=float)
        for boid in boids:
            if np.linalg.norm(boid.center - self.center) < perception:
                center_of_mass += boid.center
                count += 1
        if count > 0:
            center_of_mass /= count
            vec_to_com = center_of_mass - self.center
            if np.linalg.norm(vec_to_com) > 0:
                vec_to_com = (vec_to_com / np.linalg.norm(vec_to_com)) * max_speed
            steering = vec_to_com - self.v
            if np.linalg.norm(steering)> max_force:
                steering = (steering /np.linalg.norm(steering)) * max_force
        return steering * c_cohesion

    def separation(self, boids):
        steering = np.zeros(3, dtype=float)
        count = 0
        avg_vector = np.zeros(3, dtype=float)
        for boid in boids:
            distance = np.linalg.norm(boid.center - self.center)
            if (self.center != boid.center).all() and (distance < perception):
                diff = self.center - boid.center
                diff /= distance
                avg_vector += diff
                count += 1
        if count > 0:
            avg_vector /= count
            if np.linalg.norm(steering) > 0:
                avg_vector = (avg_vector / np.linalg.norm(steering)) * max_speed
            steering = avg_vector - self.v
            if np.linalg.norm(steering) > max_force:
                steering = (steering /np.linalg.norm(steering)) * max_force
        return steering * c_separation

class Boids:
    def __init__(self, filename, initPositions):
        self.filename = filename
        self.initPositions = initPositions
        self.objs = []

    def addGround(self):
        ground_vertices = (
            (-20,20,-10),
            (-20,-20,-10),
            (20,-20,-10),
            (20,20,-10)
        )
        glBegin(GL_QUADS)
        for vertex in ground_vertices:
            glColor3fv((1,1,1.5))
            glVertex3fv(vertex)
        glEnd()

    def addWalls(self):
        ground_vertices = (
            (-20,20,-10),
            (-20,-20,-10),
            (-20,-20,10),
            (-20,20,10)
        )
        glBegin(GL_QUADS)
        for vertex in ground_vertices:
            glColor3fv((0,3,1.5))
            glVertex3fv(vertex)
        glEnd()
        ground_vertices = (
            (20,20,-10),
            (-20,20,-10),
            (-20,20,10),
            (20,20,10)
        )
        glBegin(GL_QUADS)
        for vertex in ground_vertices:
            glColor3fv((3,0,1.5))
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

        glTranslate(0, 5, -25)
        glRotatef(-30,1,0.1,0)
        glRotatef(-30,0,0,1)
        self.addGround()
        self.addWalls()
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
            self.objs.append(Bird(self.filename, init[0], init[1], init[2]))

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        width, height = viewport
        gluPerspective(90.0, width/float(height), 1, 100.0)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_MODELVIEW)

        while True:
            # for every timestamp
            for dt in range(3000):
                # move each bird
                for bird in self.objs:
                    bird.apply_behaviour(self.objs)
                    bird.move()
                # draw
                self.drawOneFrame()

if __name__ == "__main__":
    initPositions = [[2, 7, 5], [2, -7, 5], [-2, -2, 5], [-2, 2, 5], [3, -5, 5], [5, -5, 5],
                        [2, -12, -5], [15, -12, -5], [10, -14, -5], [-12, 7, -5], [3, -15, -5], [5, -15, -5]]
    physics = Boids('bird.obj', initPositions)
    physics.draw()










