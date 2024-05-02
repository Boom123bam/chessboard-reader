import roboflow

roboflow.login()

rf = roboflow.Roboflow()
project = rf.workspace("chess-detector").project("chess-predictor")
version = project.version(10)
dataset = version.download("folder")
