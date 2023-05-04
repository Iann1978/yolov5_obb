import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

class DotaViewer:
    def __init__(self, path):
        self.path = path

    def run(self):
        # get all the images
        imgfiles = glob.glob(os.path.join(self.path, "images", "train", "*.png"))
        # get all the labels
        # labelfiles = glob.glob(os.path.join(self.path, "labelTxt", "*.txt"))
        # get the image and label file pairs
        imglabelfiles = [(x, x.replace("images", "labels").replace("png", "txt")) for x in imgfiles]

        plt.ion()
        # loop through the image and label file pairs
        for imgfile, labelfile in imglabelfiles:
            # read the image
            img = cv2.imread(imgfile)
            # read the labels
            labels = self.read_dota_labels(labelfile)
            # draw the labels
            img = self.draw_labels(img, labels)
            # show the image with plt
            plt.imshow(img)
            plt.show()
            # plt.ginput(1)
            plt.waitforbuttonpress()

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

    def draw_labels(self, img, labels):
        """
        Draw the labels on the image.
        """
        # loop through the labels
        for label in labels:
            # get the points
            points = np.array(label['points']).astype(np.int32).reshape(-1, 2)
            # get the class name
            name = label["name"]
            # draw the points
            cv2.polylines(img, [points], True, (0, 255, 0), 2)
        
            x1 = points[0][0]
            y1 = points[0][1]
            cv2.putText(img, name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

if __name__ == "__main__":
    # get the path
    path = r"./dotadata"
    # path = r"/datasets/dota2"
    # path = r"/datasets/dota_15_origin"
    # create the viewer
    viewer = DotaViewer(path)
    # run the viewer
    viewer.run()