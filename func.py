import cv2
import numpy as np
import time

# ------------- CONSTANTES ------------- #
BLOBSIZE = 416
colors = [(0,255,0),(0,0,255)]
BORDER_SIZE = 3
WHITE = (255,255,255)
BLACK = (0,0,0)
THICKNESS = 2
font = cv2.FONT_HERSHEY_TRIPLEX

# ------------- INITIALISATION ------------- #
ver__weight = 3
version_weight = "masque_v3.weights"
net = cv2.dnn.readNet("masque_v3.weights", "masque_custom.cfg")
classes = []
with open("classes.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
ratio = 2

class_ids = []
confidences = []
boxes = []
indexes = []

# ------------- CLASSE POUR LA VERSION ------------- #
class Version:
    '''
    Probablement la pire facon de résoudre le problème, mais ca marche
    En bref, il fallait récuperer la valeur de la listbox et l'utiliser pour changer le net en passant d'un fichier à l'autre !!
    Avec des objets, on peut facilement passer de l'un à l'autre en conservant des variables locales (?)
    A AMELIORER !!!!
    '''
    maListe = list()
    def __init__(self, ver):
        self.ver = ver
        Version.maListe = [self.ver]

# ------------- CLASSE AVEC METHODES ------------- #
class MesFonctions():
    def __init__(self):
        pass
    '''
    La premiere méthode renvoie une image analysée, la convertion img (np.array) --> sg.Image se fait dans l'application directement.
    La seconde prend une vidéo et en crée une nouvelle traitée dans le dossier d'origine de la vidéo.
    '''

    @staticmethod
    def init_yolo():
        v_w = Version.maListe[0]
        net = cv2.dnn.readNet(v_w, "masque_custom.cfg")
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        return net, layer_names,output_layers

    @staticmethod
    def AnalyseImg(img, frameID,net, output_layers, period=1):


        global class_ids, confidences, boxes, indexes, colors

        if frameID % period == 0:
            class_ids = []
            confidences = []
            boxes = []

            height, width, channels = img.shape
            # Detecting objects
            blob = cv2.dnn.blobFromImage(img, 0.00392, (BLOBSIZE, BLOBSIZE), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                textSize = cv2.getTextSize(label, font, 1, 2)
                textSize_width = textSize[0][0] + 5
                textSize_height = textSize[0][1] + 10
                surete = str(round(confidences[i], 2) * 100) + "%"
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)  # Main rectangle, no touchar

                cv2.rectangle(img, (x, y - 5), (x + textSize_width, y - textSize_height), color,
                                      -1)  # Label rectangle (Border)
                cv2.rectangle(img, (x + BORDER_SIZE, y - (5 + BORDER_SIZE)),
                                      (x + (textSize_width - BORDER_SIZE), y - (textSize_height - BORDER_SIZE)), BLACK,
                                      -1)  # Label rectangle (black inside)

                cv2.putText(img, label, (x + 3, y - 10), font, 1, WHITE, THICKNESS)
                cv2.putText(img, surete, (x + w - cv2.getTextSize(surete, font, 1, 2)[0][0] - 2,
                                                  y + h - cv2.getTextSize(surete, font, 1, 2)[0][1] + 10), font, 1, (0, 0, 255),
                                    THICKNESS)

        return img

    @staticmethod
    def AnalyseVid(path, ratio,net, output_layers ,period=1):


        global class_ids, confidences, boxes, indexes, colors


        calcul_temps = False


        cap = cv2.VideoCapture(path)
        print("Processing new videoooooo...")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        deb_proc = time.time()
        frameLoop = 0
        frame_id = 0
        nombre_boucle = 0
        start_time = time.time()
        _, img = cap.read()
        height, width, channels = img.shape
        writer = cv2.VideoWriter(path + '-p{0}-v{1}.avi'.format(period, ver__weight), cv2.VideoWriter_fourcc(*'DIVX'), 25, (round(width/ratio), round(height/ratio)))
        while True:
            a = time.time()
            _,img = cap.read()

            frameLoop += 1
            frame_id+=1

            if frameLoop == period and _ == True:
                nombre_boucle += 1
                height, width, channels = img.shape
                # Detecting objects

                blob = cv2.dnn.blobFromImage(img, 0.00392, (BLOBSIZE, BLOBSIZE), (0, 0, 0), True, crop=False)

                net.setInput(blob)
                outs = net.forward(output_layers)

                # Showing information on the screen
                class_ids = []
                confidences = []
                boxes = []
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:
                            # Object detected
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)

                            # Rectangle coordinates
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                frameLoop = 0

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    textSize = cv2.getTextSize(label, font, 1, 2)
                    textSize_width = textSize[0][0] + 5
                    textSize_height = textSize[0][1] + 10
                    surete = str(round(confidences[i], 2)) + "%"
                    color = colors[class_ids[i]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)  # Main rectangle, no touchar

                    cv2.rectangle(img, (x, y - 5), (x + textSize_width, y - textSize_height), color,
                                  -1)  # Label rectangle (Border)
                    cv2.rectangle(img, (x + BORDER_SIZE, y - (5 + BORDER_SIZE)),
                                  (x + (textSize_width - BORDER_SIZE), y - (textSize_height - BORDER_SIZE)), BLACK,
                                  -1)  # Label rectangle (black inside)

                    cv2.putText(img, label, (x + 3, y - 10), font, 1, WHITE, THICKNESS)
                    cv2.putText(img, surete, (x + w - cv2.getTextSize(surete, font, 1, 2)[0][0] - 2,
                                              y + h - cv2.getTextSize(surete, font, 1, 2)[0][1] + 10), font, 1, (0, 0, 255),
                                THICKNESS)

            elapsed_time = time.time() - start_time
            fps = round(frame_id/elapsed_time, 2)
            cv2.putText(img, "FPS: "+ str(fps), (10, 50), font, 2, (0, 0, 0), 1)

            if calcul_temps == False and nombre_boucle == 1:
                calcul_temps = True
                prediction_temps = ((time.time()-a)/period * total_frames)
                print("Il faudra ~{} à {} minutes".format(prediction_temps//60,prediction_temps//60 + 1))


            if _ == True:
                fin = cv2.resize(img,(round(width/ratio), round(height/ratio)))
                writer.write(fin)

            elif  _ == False:
                break

        writer.release()
        cap.release()
        cv2.destroyAllWindows()
        print("Traitement fini. Temps passé : {} sec".format(round(time.time()-deb_proc,2)))

if __name__ == "__main__":
    print('Develeopofpofpzo')