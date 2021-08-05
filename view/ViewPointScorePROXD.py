from vedo import Plotter, Points, Point, load, interactive

from vedo.utils import flatten

class ViewPointScorePROXD():

    def __init__(self, controler, file_env):
        self.vp = Plotter(shape=(1,2), title="Scores", bg="white", size=(1600, 800), axes=1)

        self.controler = controler
        self.vedo_file_env = load(file_env)
        self.vedo_file_env.lighting(ambient=0.8, diffuse=0.2, specular=0.1, specularPower=1, specularColor=(1, 1, 1))

        self.point_clouds=[]
        self.right_side_elements=[self.vedo_file_env]

    def add_point_cloud(self, np_points, np_scores, r=5, at=0):
        pts = Points(np_points, r=r)
        pts.cellColors(np_scores, cmap='jet', vmin=0, vmax=self.controler.max_limit_score)
        pts.addScalarBar(pos=(0.8, 0.25), nlabels=5, title="PV alignment distance", titleFontSize=10)
        self.point_clouds.append(pts)

    def add_vedo_element(self, vedo_element, at):
        # if at ==0:
        #     self.left_side_elements.append(vedo_element)
        #
        # else:
        #     self.right_side_elements.append(vedo_element)
        self.vp.add(vedo_element, at=at)

    def show(self):
        self.vp.mouseRightClickFunction = self.on_right_click
        self.vp.show(flatten([self.vedo_file_env,self.point_clouds]), at=0)
        self.vp.show(self.vedo_file_env, at=1)
        # self.vp.show() # this permit independent visualization
        interactive()

    def on_right_click(self, mesh):
        if mesh.picked3d is not None:
            np_point, best_angle = self.controler.get_data_from_nearest_point_to(mesh.picked3d)
            self.controler.optimize_best_scored_position(np_point, best_angle)
            # self.show()
