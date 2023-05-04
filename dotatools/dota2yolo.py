import cv2
import os
import shutil
import numpy as np
import glob

class Dota2Yolo:
    classnames_v1_5 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                    'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']

                
    def __init__(self, dotapath, yolopath, obb:bool=False) -> None:
        """
        """
        self.dotapath = dotapath
        self.yolopath = yolopath
        # self.dotaimgpath = os.path.join(self.dotapath, "images")
        # self.dotalabelpath = os.path.join(self.dotapath, "labelTxt")
        # self.yoloimgpath = os.path.join(self.yolopath, "images")
        # self.yololabelpath = os.path.join(self.yolopath, "labels")
        # self.dotaimglist = os.listdir(self.dotaimgpath)
        # self.dotalabellist = os.listdir(self.dotalabelpath)
        self.obb = obb

    
    def classname_to_id(self, classname):
        """
        Convert the class name to id.
        """
        return self.classnames_v1_5.index(classname)

    def convert_one_label(self, dotaimgfile, dotalabelfile, yololabelfile):
        """
        Convert one label file from dota to yolo format.
        """

        # read the image file
        img = cv2.imread(dotaimgfile)
        imgshape = img.shape

        # read one label file
        dota_labels = self.read_dota_labels(dotalabelfile)
        for onelabel in dota_labels:
            # convert to yolo format
            onelabel = self.convert_one_label_to_yolo_obb(onelabel, imgshape) if self.obb else self.convert_one_label_to_yolo(onelabel, imgshape)
            self.append_one_to_yolo_lable_file(yololabelfile, onelabel)

    def append_one_to_yolo_lable_file(self, yololabelfile, onelabel):
        """
        Append one label to the yolo label file.
        """
        with open(yololabelfile, "a") as f:
            f.write(str(onelabel["id"]) + " " + " ".join(
                [str(x) for x in onelabel["points"]]))
            f.write("\n")
    
    def read_dota_labels(self, dotalabelfile):
        """
        Read one label file from dota format.
        """
        dota_labels = []
        # read the label file
        with open(dotalabelfile, "r") as f:
            for line in f:
                words = line.strip().split(" ")
                if len(words) < 9:
                    continue
                onelabel = {}
                onelabel["name"] = words[8]
                onelabel["points"] = [[float(words[0]), float(words[1])],
                                      [float(words[2]), float(words[3])],
                                      [float(words[4]), float(words[5])],
                                      [float(words[6]), float(words[7])]]
                dota_labels.append(onelabel)
        return dota_labels
 
    def convert_one_label_to_yolo(self, onelabel, imgshape):
        """
        Convert one label from dota to yolo format.
        """
        # get the points
        points = onelabel["points"]
        # get the name
        name = onelabel["name"]
        # get the id
        id = self.classname_to_id(name)
        # get the width and height
        width = max([x[0] for x in points]) - min([x[0] for x in points])
        height = max([x[1] for x in points]) - min([x[1] for x in points])
        # get the center
        center = [(max([x[0] for x in points]) + min([x[0] for x in points])) / 2,
                    (max([x[1] for x in points]) + min([x[1] for x in points])) / 2]
        # get the yolo points
        yolo_points = [center[0] / imgshape[1], center[1] / imgshape[0],
                        width / imgshape[1], height / imgshape[0]]
        # return the yolo label
        return {"id": id, "points": yolo_points}


    def convert_one_label_to_yolo_obb(self, onelabel, imgshape):
        """
        Convert one label from dota to yolo(obb) format.
        """
        # get the points
        points = onelabel["points"]
        # get the name
        name = onelabel["name"]
        # get the id
        id = self.classname_to_id(name)
        # get the bbox
        bbox = cv2.minAreaRect(np.array(points).astype(np.float32))
        # get the width and height
        width = bbox[1][0]
        height = bbox[1][1]
        # get the center
        center = bbox[0]
        # get the angle
        angle = int(bbox[2])
        # get the yolo points with angle
        yolo_points = [center[0] / imgshape[1], center[1] / imgshape[0],
                        width / imgshape[1], height / imgshape[0], angle]
        # return the yolo label
        return {"id": id, "points": yolo_points}
    
    def run(self):
        """
        Run the convert.
        """

        dotatrainimgpath = os.path.join(self.dotapath, "images", "train" )
        dotatrainlabpath = os.path.join(self.dotapath, "labels", "train")
        dotavalimgpath = os.path.join(self.dotapath, "images", "val")
        dotavallabpath = os.path.join(self.dotapath, "labels", "val")

        yolotrainimgpath = os.path.join(self.yolopath, "images", "train")
        yolotrainlabpath = os.path.join(self.yolopath, "labels", "train")
        yolovalimgpath = os.path.join(self.yolopath, "images", "val")
        yolovallabpath = os.path.join(self.yolopath, "labels", "val")



        # delete the yolo path
        if os.path.exists(self.yolopath):
            shutil.rmtree(self.yolopath)
        # create the yolo path
        os.makedirs(yolotrainimgpath)
        os.makedirs(yolotrainlabpath)
        os.makedirs(yolovalimgpath)
        os.makedirs(yolovallabpath)

        # get the dota image list
        dotatrainimgfiles = glob.glob(os.path.join(dotatrainimgpath, "*.png"))
        dotavalimgfiles = glob.glob(os.path.join(dotavalimgpath, "*.png"))
        dotaimgfiles = dotatrainimgfiles + dotavalimgfiles

        # convert the label files
        for dotaimgfile in dotaimgfiles:
            dotalabelfile = dotaimgfile.replace("/images/","/labels/").replace(".png", ".txt")
            yoloimgfile = dotaimgfile.replace(self.dotapath, self.yolopath)
            yololabelfile = dotalabelfile.replace(self.dotapath, self.yolopath)
            # yololabelfile = dotaimgfile.replace(".png", ".txt")
            # dotaimgfile = os.path.join(self.dotapath, dotaimgfile)
            # dotalabelfile = os.path.join(self.dotapath, dotalabelfile)
            # yololabelfile = os.path.join(self.yolopath, yololabelfile)
            # copy the image file
            shutil.copy(dotaimgfile, yoloimgfile)
            # convert the label file
            self.convert_one_label(dotaimgfile, dotalabelfile, yololabelfile)



        # convert the label files
        # for dotaimgfile in self.dotaimglist:
        #     dotalabelfile = dotaimgfile.replace(".png", ".txt")
        #     yololabelfile = dotaimgfile.replace(".png", ".txt")
        #     dotaimgfile = os.path.join(self.dotaimgpath, dotaimgfile)
        #     dotalabelfile = os.path.join(self.dotalabelpath, dotalabelfile)
        #     yololabelfile = os.path.join(self.yololabelpath, yololabelfile)
        #     # copy the image file
        #     shutil.copy(dotaimgfile, self.yoloimgpath)
        #     # convert the label file
        #     self.convert_one_label(dotaimgfile, dotalabelfile, yololabelfile)




# judge if the file is the main file

if __name__ == "__main__":
    # dotapath = "./dotadata"
    # yolopath = "./yolodata"
    dotapath = "/datasets/dota_15_origin"
    yolopath = "/datasets/dota_15_yolo"
    d2y = Dota2Yolo(dotapath, yolopath, obb=True)
    d2y.run()