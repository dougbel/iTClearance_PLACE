import vedo
import random
import numpy as np
import math

class Selector:

    def __init__(self, trimesh_scene, scene_min_x, scene_max_x, scene_min_y, scene_max_y, r=2.0):
        # this step is necesary to keep color of scene
        trimesh_scene.visual.face_colors = trimesh_scene.visual.face_colors
        self.activated = False
        self.scene_min_x = scene_min_x
        self.scene_max_x = scene_max_x
        self.scene_min_y = scene_min_y
        self.scene_max_y = scene_max_y
        self.r = r

        self.txt_enable = vedo.Text2D('Left click selection enabled ("c")', pos='bottom-right', c='steelblue', bg='black',
                            font='ImpactLabel', alpha=1)
        self.txt_disable = vedo.Text2D('Left click selection disabled ("c")', pos='bottom-right', c='darkred', bg='black',
                             font='ImpactLabel', alpha=1)
        self.txt_invalid = vedo.Text2D(f'Invalid point, necessary a distance of {r/2} from  scene limits', pos='top-right', c='darkred',
                                       bg='black', font='ImpactLabel', alpha=1)
        self.vedo_env = vedo.utils.trimesh2vtk(trimesh_scene).lighting('ambient')

        self.vp = None

        self.vedo_testing_point = None
        self.np_testing_point = None
        self.is_valid_point = False

    def select_point_to_test(self):
        self.vp = vedo.Plotter(bg="white", axes=9)
        self.vp.mouseLeftClickFunction = self.on_left_click
        self.vp.keyPressFunction = self.on_key_press
        self.vp.show(self.vedo_env, self.txt_disable)
        self.vp.close()
        return self.np_testing_point

    def on_left_click(self, mesh):
        if self.activated:
            if self.vedo_testing_point is not None:
                self.vp.clear(self.vedo_testing_point)

            self.np_testing_point = np.asarray(mesh.picked3d)

            if self.scene_min_x + self.r / 2 <= self.np_testing_point[0] <=self.scene_max_x - self.r / 2 and self.scene_min_y + self.r / 2 <= self.np_testing_point[1]<= self.scene_max_y - self.r / 2:
                self.vedo_testing_point = vedo.Point(self.np_testing_point, c='green')
                self.vp.clear(self.txt_invalid)
                self.is_valid_point = True
            else:
                self.vedo_testing_point = vedo.Point(self.np_testing_point, c='red')
                self.vp.add(self.txt_invalid)
                self.is_valid_point = False

            self.vp.add(self.vedo_testing_point)



    def on_key_press(self, key):
        if key == 'c':
            self.activated = not self.activated
            if self.activated:
                self.vp.add(self.txt_enable)
                self.vp.clear(self.txt_disable)
            else:
                self.vp.clear(self.txt_enable)
                self.vp.add(self.txt_disable)








if __name__ == '__main__':
    import os
    import trimesh
    dataset_prox_dir = "/home/dougbel/Documents/UoB/5th_semestre/to_test/place_comparisson/data/datasets/prox/scenes"
    scene_name = 'MPH16.ply'

    thimesh_scn = trimesh.load(os.path.join(dataset_prox_dir, scene_name))
    bb_vertices = thimesh_scn.bounding_box.vertices

    scene_max_x, scene_max_y = bb_vertices[:, :2].max(axis=0)
    scene_min_x, scene_min_y = bb_vertices[:, :2].min(axis=0)


    a = Selector(thimesh_scn, scene_min_x, scene_max_x, scene_min_y, scene_max_y)
    print(a.select_point_to_test())

    print(a.select_point_to_test())