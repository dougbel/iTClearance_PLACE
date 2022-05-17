import time

from vedo import Plotter, Points, Point, load, interactive

from vedo.utils import flatten, vtk2trimesh


class ViewExtractSamplesPointScorePROXD_test():

    def __init__(self, controler, file_env):
        self.vp = Plotter(shape=(1,2), title="Scores", bg="gray", size=(1600, 800), axes=1)

        self.controler = controler
        self.vedo_file_env = load(file_env)
        self.vedo_file_env.lighting(ambient=0.8, diffuse=0.2, specular=0.1, specularPower=1, specularColor=(1, 1, 1))
        self.vedo_file_env.backFaceCulling(False)

        self.point_clouds=[]
        self.right_side_elements=[self.vedo_file_env]
        self.started = False
        self.saving_outputs=False

    def add_point_cloud(self, np_points, np_scores, r=5, at=0):
        pts = Points(np_points, r=r)
        pts.cellColors(np_scores, cmap='jet', vmin=0, vmax=self.controler.max_limit_score)
        pts.addScalarBar(pos=(0.8, 0.25), nlabels=5, title="PV alignment distance", titleFontSize=10)
        self.point_clouds.append(pts)

    def add_vedo_element(self, vedo_element, at):
        self.vp.add(vedo_element, at=at)

    def start(self, saving_outputs=False):
        self.saving_outputs = saving_outputs
        self.started = True
        self.vp.mouseRightClickFunction = self.on_right_click
        self.vp.show(flatten([self.vedo_file_env,self.point_clouds]), at=0, axes=4)
        self.vp.show(self.vedo_file_env, at=1)
        # self.vp.show() # this permit independent visualization
        interactive()

    def on_right_click(self, mesh):
        import numpy as np
        if mesh.picked3d is not None:
            np_point, best_angle = self.controler.get_data_from_nearest_point_to(mesh.picked3d)
            body_trimesh_optim, __= self.controler.optimize_best_scored_position(np_point, best_angle)
            if self.saving_outputs:
                timestamp = int(time.time())
                body_trimesh_optim.export(f"{self.controler.affordance_name}_{timestamp}_optim.ply", "ply")
                vtk_object = load(self.controler.tester.it_descriptor.object_filename())
                vtk_object.c([150, 150, 0])
                vtk_object.rotate(best_angle, axis=(0, 0, 1), rad=True)
                vtk_object.pos(x=np_point[0], y=np_point[1], z=np_point[2])
                body_trimesh = vtk2trimesh(vtk_object)
                body_trimesh.export(f"{self.controler.affordance_name}_{timestamp}.ply", "ply")


