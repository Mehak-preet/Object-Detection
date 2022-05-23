from Detector import *
import os


def main():
    #videoPath = "C:\\Users\\HP\\OneDrive\\Desktop\Object_Detection\\test\\traffic.mp4"
    videoPath=0  #for using webcam

    configPath = os.path.join("model", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model", "frozen_inference_graph.pb")
    classesPath = os.path.join("model", "coco.names")

    detector = Detector(videoPath, configPath, modelPath, classesPath)
    detector.onVideo()


if __name__ == '__main__':
    main()
