import PySimpleGUIQt as sg
import PySimpleGUI as sg2
import cv2
import numpy as np
import func as yf
import ctypes

# ------------- Version par défaut (la meilleure, un peu mieux que la 4/5) ------------- #
version = yf.Version("masque_v3.weights")


def main():

    global version


    weights = ['masque_v{}.weights'.format(i) for i in range(1,6)]
    net, layer_names, output_layers = yf.MesFonctions.init_yolo()

    user32 = ctypes.windll.user32
    screensize = round(user32.GetSystemMetrics(0)/2), round(user32.GetSystemMetrics(1)/2)

    sg.theme('Kayak')

    layout = [ [sg.Button('Detection sur Webcam', size=(22, 1), font='Helvetica 14'),
                sg.Button('Analyser une vidéo', size=(20, 1), font='Any 14'),
                sg.Text("Inserez la fréquence de calcul en nombre d'images"),
                sg.Input('1',key='-period_val-'),
                sg.Button('OK', size=(10,1), font='Any 14', button_color=('white','green'))],
                [sg.Listbox(values=weights, size=(20,len(weights)), enable_events=True, key='-LIST-'), sg.Image(filename='logo.png', key='image') ],
               [sg.Text('Version du fichier weight : 3', key='-VERSION-',font='Helvetica 26' ), sg.Stretch(), sg.Button('i', size=(3,1))]
            ]

    window = sg.Window('Projet port du masque S4', layout, size=screensize, finalize=True, element_justification='c')


    cap = cv2.VideoCapture(0)
    recording = False

    frameID = 0
    period = 1

    while True:
        event, values = window.read(timeout=20)

        if event == sg.WIN_CLOSED:
            break

        elif event == 'OK':

            try:
                recording = False
                period = int(values['-period_val-'])
                if values['-LIST-']:
                    weight = values['-LIST-'][0]
                    window['-VERSION-'].update('Version du fichier weight :  {}'.format(weights.index(weight) + 1))
                    version = yf.Version(weight)
                    net, layer_names,output_layers =  yf.MesFonctions.init_yolo()
                    sg.popup("L'analyse se fera toutes les {} images.\nLa version du fichier weight est : {}".format(period,values['-LIST-'][0]), title='OK', keep_on_top=True)

                else:
                    sg.popup("L'analyse se fera toutes les {} images.\nLa version du fichier weight est : 'masque_v3.weights'".format(period), title='OK', keep_on_top=True)
                recording = False
                window['image'].update('logo.png')
            except ValueError:
                sg.Popup('Entrez un entier !', keep_on_top=True)

        elif event == 'Detection sur Webcam':
            recording = True

        elif event == 'Analyser une vidéo':
            path = sg.popup_get_file('Document to open')
            if path is not None:
                try:
                    yf.MesFonctions.AnalyseVid(path, 3, net, output_layers, period)
                    sg.popup('Fini !', keep_on_top=True)
                except UnboundLocalError:
                    sg.popup('Veuillez rentrer un fichier formel.', keep_on_top=True)
                    print(path)
                except AttributeError:
                    sg.popup('Veuillez rentrer un fichier formel.', keep_on_top=True)
                    print(path)
            else:
                pass

        elif event == 'i':
            sg.Popup("Developpé par Pierre Pinon et Joshua VHB pour le projet\nPeiP2 - S4 de l'année 2020-2021!!", title='Infos' )


        if recording:
            frameID += 1
            ret, frame = cap.read()
            yf.MesFonctions.AnalyseImg(frame, frameID,net, output_layers,period)
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
            window['image'].update(data=imgbytes)


    window.close()


if __name__ == '__main__':
    main()

