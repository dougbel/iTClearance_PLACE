import logging

from vedo import Plotter, write, merge, load
import os

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    # data_dir = "/media/dougbel/Tezcatlipoca/datasets_place/prox"
    # data_dir = "/media/dougbel/Tezcatlipoca/datasets_place/mp3d"
    data_dir = "/media/dougbel/Tezcatlipoca/datasets_place/replica_v1"

    for file_name in os.listdir( os.path.join(data_dir,"scenes")):
        vp = Plotter(bg="white")
        e = load(os.path.join(data_dir, "scenes", file_name))
        e.lighting(ambient=0.8, diffuse=0.2, specular=0.1, specularPower=1, specularColor=(1, 1, 1))
        vp.add(e)
        fine = load(os.path.join(data_dir, "scenes_filled", "filler_fine_bubbles_" + file_name))
        fine.color("blue").lighting(ambient=0.8, diffuse=0.2, specular=0.1, specularPower=1, specularColor=(1, 1, 1))
        vp.add(fine)
        gross = load(os.path.join(data_dir, "scenes_filled", "filler_gross_bubbles_" + file_name))
        gross.color("orange").lighting(ambient=0.8, diffuse=0.2, specular=0.1, specularPower=1, specularColor=(1, 1, 1))
        vp.add(gross)
        filler = load(os.path.join(data_dir, "scenes_filled", "filler_floor_holes_" + file_name))
        filler.color("green").lighting(ambient=0.8, diffuse=0.2, specular=0.1, specularPower=1, specularColor=(1, 1, 1))
        vp.add(filler)

        vp.show(title=file_name)
