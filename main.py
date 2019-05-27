#!/usr/bin/env python
# -*- coding: utf-8 -*-


from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.label import Label

import cv2
import numpy as np


class CameraApp(App):

    def build(self):
        self.img1 = Image(source='logoCL.jpg')  # cria uma imagem onde depois iremos inserir a imagem da camera
        layout = BoxLayout(orientation='vertical')  # |aqui criamos um layout  vertical
        layout.add_widget(self.img1)

        self.capture = cv2.VideoCapture(0)
        ret, frame = self.capture.read()

        Clock.schedule_interval(self.atualizaImagem, 1.0 / 30.0)
        return layout

    def atualizaImagem(self, dt):
        ret, frame = self.capture.read()

        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        buf1 = cv2.flip(frame, 0)
        #buf1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        buf = buf1.tostring()

        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        self.img1.texture = texture1

if __name__ == '__main__':
    CameraApp().run()