import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from PIL import Image, ImageDraw, ImageFont
import math


class YoloViewer:
    def __init__(self, path, classes, trainpath='train', ext='png', obb=False, pil=False):
        self.path = path
        self.trainpath = trainpath
        self.ext = ext
        self.classes = self.verify_classes(classes)
        self.obb = obb
        self.pil = pil
    
    def verify_classes(self, classes):
        """
        Verify that the classes are in the label file.
        """
        # judge if the classes is array 
        if isinstance(classes, list):
            # get the class names
            classnames = classes
        # judge if the classes is filename
        elif os.path.isfile(classes):
            # check if the classes file exists
            assert os.path.isfile(classes), "classes file {} does not exist".format(classes)
            # judge if the classes is yaml file
            if classes.endswith(".yaml"):
                # read the yaml file
                import yaml
                classnames = yaml.load(open(classes, "r"), Loader=yaml.FullLoader)
                classnames = classnames["names"]
            else:
                classnames = [x.strip() for x in open(classes).readlines()]
        # judge if the classes is string
        elif isinstance(classes, str):
            # execute the string as python code
            classnames = eval(classes)

        return classnames
        



    
    def run(self):
        # get all the images
        imgfiles = glob.glob(os.path.join(self.path, "images", self.trainpath, "*."+self.ext))
        # get all the labels
        labelfiles = glob.glob(os.path.join(self.path, "labels",self.trainpath, "*."+self.ext))
        # get the image and label file pairs
        imglabelfiles = [(x, x.replace("images", "labels").replace(self.ext, "txt")) for x in imgfiles]

        plt.ion()
        # loop through the image and label file pairs
        for imgfile, labelfile in imglabelfiles:
            # read the image
            img = Image.open(imgfile) if self.pil else cv2.imread(imgfile)
            # read the labels
            labels = self.read_yolo_labels(labelfile)
            # draw the labels
            img = self.draw_yolo_labels(img, labels) # if not self.obb else self.draw_yolo_labels_obb(img, labels)
            # show the image with plt
            plt.imshow(img)
            plt.show()
            plt.ginput(1)

    def read_yolo_labels(self, yololabelfile):
        """
        Read one label file from yolo format.
        """
        yolo_labels = []
        # read the label file
        with open(yololabelfile, "r") as f:
            for line in f:
                words = line.strip().split(" ")
                if len(words) < 5:
                    continue
                onelabel = {}
                onelabel["id"] = words[0]
                onelabel["name"] = self.classes[int(words[0])]
                onelabel["points"] = [[float(words[1]), float(words[2])],
                                    [float(words[3]), float(words[4])]]
                if self.obb:
                    onelabel["points"].append([float(words[5])])
                yolo_labels.append(onelabel)
        return yolo_labels
    
    def draw_yolo_labels(self, img, labels):
        """
        Draw the labels on the image.
        The labels are in yolo format.
        points: [[cx, cy], [w, h]]
        """
        for label in labels:
            # get the points
            points = label["points"]
            # get the class name
            name = label["name"]

            width = img.size[0] if self.pil else img.shape[1]
            height = img.size[1] if self.pil else img.shape[0]

            # get center scaled by image width and height
            cx = points[0][0] * width
            cy = points[0][1] * height
            # get width and height scaled by image width and height
            w = points[1][0] * width
            h = points[1][1] * height
            # get angle
            theta = 0.0 if len(points) < 3 else points[2][0]

            # get obb ((cx,cy), (w,h), theta)
            obb = ((cx, cy), (w, h), theta)

            img = self.draw_obb_with_pil(img, obb, name) if self.pil else self.draw_obb_with_cv2(img, obb, name)



            # get the top left corner
            x1 = cx - w / 2
            y1 = cy - h / 2

            # draw name at the top left corner
            # cv2.putText(img, name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img
    
    # draw the labels on the image using opencv
    def draw_obb_with_cv2(self, img, obb, name):
        (cx, cy), (w, h), theta = obb
        # make tuple bbox through calling cv2.boxPoints(rect)
        bbox = cv2.boxPoints(((cx, cy), (w, h), theta)).astype(np.int32)

        # draw the bbox
        cv2.drawContours(img, [bbox], 0, (0, 255, 0), 2)

        x1 = cx - w / 2
        y1 = cy - h / 2

        # draw name at the top left corner
        cv2.putText(img, name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img





    # draw an obb((cx,cy), (w,h), theta) on image using ImageDraw
    def draw_obb_with_pil(self, img, obb, name):
        """
        Draw an obb((cx,cy), (w,h), angle) on image using ImageDraw.
        """
        (cx, cy), (w, h), theta = obb

        bbox = cv2.boxPoints(((cx, cy), (w, h), theta)).astype(np.int32)



        corners = [
            (bbox[0][0], bbox[0][1]),
            (bbox[1][0], bbox[1][1]),
            (bbox[2][0], bbox[2][1]),
            (bbox[3][0], bbox[3][1]),
        ]
        # Draw the bounding box on the image

        draw = ImageDraw.Draw(img)        
        draw.polygon(corners, outline=(255, 0, 0))

        x1 = cx - w / 2
        y1 = cy - h / 2

        # draw name at the top left corner using ImageDraw
        draw.text((x1, y1), name, fill=(255, 0, 0))

        return img



            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YoloViewer")
    parser.add_argument("--path", type=str, default="/datasets/coco128")
    args = parser.parse_args()


    # viewer = YoloViewer(args.path,trainpath='train2017',classes='data/coco128.yaml',ext='jpg')
    # viewer.run()


    
    classnames_v1_5 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']

    path = r"./yolodata"
    # viewer = YoloViewer(path,classes=classnames_v1_5, ext='png', obb=True, pil=True)
    # viewer.run()

    # path = r"/datasets/coco128"
    # viewer = YoloViewer(path,classes='data/coco128.yaml', trainpath='train2017', ext='jpg', obb=False, pil=True)
    # viewer.run()

    path = r"./yolodata"
    # path = r"/datasets/dota_15_yolo"
    viewer = YoloViewer(path,classes=classnames_v1_5, ext='png', obb=True, pil=True)
    viewer.run()
