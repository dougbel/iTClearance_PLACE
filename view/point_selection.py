import numpy as np
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
from vedo import Plotter, Points, Point, load


class ViewPointSelection:

    def __init__(self, controler, file_env):
        self.vp = Plotter(verbose=0, title="Scores", bg="white", size=(1200, 800))

        self.controler = controler
        self.vedo_file_env = load(file_env)

        self.vp.add(self.vedo_file_env)

    def add_point_cloud(self, np_points, np_scores, r=5):
        pts = Points(np_points, r=r)
        pts.cellColors(np_scores, cmap='jet', vmin=0, vmax=self.controler.max_limit_score)
        pts.addScalarBar(pos=(0.8, 0.25), nlabels=5, title="PV alignment distance", titleFontSize=10)
        self.vp.add(pts)

    def show(self):
        self.vp.mouseRightClickFunction = self.on_right_click
        self.vp.mouseMiddleClickFunction = self.on_middle_click
        self.vp.show()

    def on_right_click(self, mesh):
        if mesh.picked3d is not None:
            self.controler.get_data_from_nearest_point_to(mesh.picked3d)

    def on_middle_click(self, mesh):
        if mesh.picked3d is not None:
            self.controler.remove_body(mesh)