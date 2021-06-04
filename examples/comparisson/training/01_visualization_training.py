from viewer.training.ControlTrainings import CtrlTrainingsVisualizer

if __name__ == "__main__":
    ctrl = CtrlTrainingsVisualizer()
    ctrl.ui.line_descriptors.setText("/media/dougbel/Tezcatlipoca/PLACE_trainings/config/descriptors_repository")
    ctrl.start()