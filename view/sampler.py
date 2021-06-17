import numpy as np
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
from vedo import Plotter, Points, Point, load


class ViewSampler:

    def __init__(self, controler, file_env):
        self.vp = Plotter(verbose=0, title="Scores", bg="white", size=(1200, 800))

        self.controler = controler
        self.vedo_file_env = load(file_env)

        self.vp.add(self.vedo_file_env)

    def add_point_cloud(self, np_points, np_scores, r=5):
        pts = Points(np_points, r=r)
        pts.cellColors(np_scores, cmap='jet_r', vmin=0, vmax=1)
        pts.addScalarBar(c='jet_r', nlabels=5, pos=(0.8, 0.25))
        self.vp.add(pts)

    def show(self):
        self.vp.show()

