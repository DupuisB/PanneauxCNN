import random

import network
from loaders.mnist_loader import *
from loaders.pano_loader import *
from utils.activation_functions import *
from layers import *
from utils.cost_functions import *

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog as fd
from PIL import ImageTk, Image


class Software(object):

    def __init__(self, network, loader, background='#FFEFDB', size='1180x680', windows_title='CNN - Dupuis Benjamin'):
        self.loader = loader
        self.net = network
        self.loader = loader
        self.data = None

        self.title = windows_title
        self.back = background
        self.size = size
        self.font = ('arial', 11, 'normal')

        self.root = tk.Tk()

        self.root.geometry(self.size)
        self.root.configure(background=self.back)
        self.root.title(self.title)
        self.root['padx'] = 10
        self.root['pady'] = 10

        self.hauteurs = [20]

        # Load/Save frame
        self.networkFrame = tk.LabelFrame(self.root, text="Choose Network", relief=tk.RIDGE, bg=self.back)
        self.networkFrame.grid(row=1, column=1)

        tk.Label(self.networkFrame, text="Network Name", font=self.font, bg=self.back).grid(row=1, column=1, padx=10,
                                                                                            pady=10)
        self.networkname = tk.Entry(self.networkFrame)
        self.networkname.insert(0, 'test')
        self.networkname.grid(row=1, column=2, padx=10, pady=10)

        tk.Label(self.networkFrame, text="Fonction Cout", font=self.font, bg=self.back).grid(row=2, column=1,
                                                                                             padx=10, pady=10)
        self.cost_func = ttk.Combobox(self.networkFrame, values=['Quadratique', 'Cross-entropy'])
        self.cost_func.insert(0, 'Quadratique')
        self.cost_func.grid(row=2, column=2, padx=10, pady=10)

        tk.Label(self.networkFrame, text="Data Loader", font=self.font, bg=self.back).grid(row=3, column=1, padx=10,
                                                                                           pady=10)
        self.loaderTK = tk.Entry(self.networkFrame)
        self.loaderTK.insert(0, loader.__name__)
        self.loaderTK.grid(row=3, column=2, padx=10, pady=10)

        self.save = tk.Button(self.networkFrame, text='Save as name', bg=self.back, font=self.font, command=self.save)
        self.save.grid(row=4, column=1, padx=10, pady=10)

        self.load = tk.Button(self.networkFrame, text='Load by name', bg=self.back, font=self.font, command=self.load)
        self.load.grid(row=4, column=2, padx=10, pady=10)

        self.networkTK = tk.Text(self.networkFrame, bg='white', font=self.font, width=30, height=10)
        self.networkTK.grid(row=5, column=2, padx=10, pady=10)

        self.createTK = tk.Button(self.networkFrame, text='Create', bg=self.back, font=self.font, command=self.create)
        self.createTK.grid(row=5, column=1, padx=10, pady=10)

        # Instructions
        self.text = tk.LabelFrame(self.root, text="Instructions", relief=tk.RIDGE, bg=self.back)
        self.text.grid(row=2, column=1)
        tk.Label(self.text, text="Couches:\n\n"
                                 "Convolution, Dense, Reshape, Softmax\n\n"
                                 "", font=self.font, bg=self.back).grid(row=4, column=1,
                                                                        pady=10, padx=10)

        # Settings frame (parametres d'entrainement)
        self.settingsFrame = tk.LabelFrame(self.root, text="Paramètres entrainement", relief=tk.RIDGE, bg=self.back)
        self.settingsFrame.grid(row=1, column=2)

        tk.Label(self.settingsFrame, text="Nombre epochs", font=self.font, bg=self.back).grid(row=1, column=1,
                                                                                              padx=10, pady=10)
        self.Epochs = tk.Entry(self.settingsFrame)
        self.Epochs.insert(0, '1')
        self.Epochs.grid(row=1, column=2, padx=10, pady=10)

        tk.Label(self.settingsFrame, text="Taille mini-batch", font=self.font, bg=self.back).grid(row=2, column=1,
                                                                                                  padx=10, pady=10)
        self.mini_size = tk.Entry(self.settingsFrame)
        self.mini_size.insert(0, '15')
        self.mini_size.grid(row=2, column=2, padx=10, pady=10)

        tk.Label(self.settingsFrame, text="Taux d'apprentissage", font=self.font, bg=self.back).grid(row=3,
                                                                                                     column=1,
                                                                                                     padx=10,
                                                                                                     pady=10)
        self.eta = tk.Entry(self.settingsFrame)
        self.eta.insert(0, '3.0')
        self.eta.grid(row=3, column=2, padx=10, pady=10)

        # Entrainement/Evaluation frame
        self.trainFrame = tk.LabelFrame(self.root, text="Entrainement/Evaluation", relief=tk.RIDGE, bg=self.back)
        self.trainFrame.grid(row=2, column=2)

        tk.Label(self.trainFrame, text="Quel resultats (graph)?\n texte pour l'instant", font=self.font,
                 bg=self.back).grid(row=1, column=1, padx=10, pady=10)

        self.eval_acc, self.train_acc = tk.BooleanVar(), tk.BooleanVar()

        self.eval_acc_button = tk.Checkbutton(self.trainFrame, text='Evaluation Accuracy', bg=self.back,
                                              variable=self.eval_acc)
        self.eval_acc_button.grid(row=2, column=2, padx=10, pady=10)

        self.train_acc_button = tk.Checkbutton(self.trainFrame, text='Training Accuracy', bg=self.back,
                                               variable=self.train_acc)
        self.train_acc_button.grid(row=4, column=2, padx=10, pady=10)

        self.train_button = tk.Button(self.trainFrame, text='Train Network', bg=self.back, font=self.font,
                                      command=self.train)
        self.train_button.grid(row=2, column=1, padx=10, pady=10)

        self.eval_train_button = tk.Button(self.trainFrame, text='Eval training', bg=self.back, font=self.font,
                                           command=self.eval_train)
        self.eval_train_button.grid(row=4, column=1, padx=10, pady=10)

        self.eval_test_button = tk.Button(self.trainFrame, text='Eval testing', bg=self.back, font=self.font,
                                          command=self.eval_test)
        self.eval_test_button.grid(row=5, column=1, padx=10, pady=10)


        # Image frame
        self.imageFrame = tk.LabelFrame(self.root, text="Image", relief=tk.RIDGE, bg=self.back)
        self.imageFrame.grid(row=1, column=3, padx=5)

        self.image = Image.open("C:/Users/benyo/PycharmProjects/PanneauxCNN/data/EuDataset/classes/001.ppm").resize(
            (224, 224), resample=0)
        self.image = self.image.convert('RGB')
        self.imgDisplayed = ImageTk.PhotoImage(image=self.image)
        self.imageTK = tk.Label(self.imageFrame, image=self.imgDisplayed)
        self.imageTK.grid(row=1, column=1, padx=5)

        # Output frame (sortie, / resultat)
        self.outputFrame = tk.LabelFrame(self.root, text="Output", relief=tk.RIDGE, bg=self.back)
        self.outputFrame.grid(row=2, column=3)
        self.output = tk.Label(self.outputFrame, text='None')
        self.output.grid(row=1, column=1)

        self.eval_train_button = tk.Button(self.imageFrame, text='Feedforward Image', bg=self.back, font=self.font,
                                           command=self.feedforward)
        self.eval_train_button.grid(row=4, column=1, padx=10, pady=10)

        self.loadImage = tk.Button(self.imageFrame, text='Load Image', bg=self.back, font=self.font,
                                   command=self.getImage)
        self.loadImage.grid(row=3, column=1, padx=10, pady=10)

        self.randomImage = tk.Button(self.imageFrame, text='Load random', bg=self.back, font=self.font,
                                     command=self.loadRandom)
        self.randomImage.grid(row=2, column=1, padx=10, pady=10)

        # Console test
        self.consoleFrame = tk.LabelFrame(self.root, text="Console", relief=tk.RIDGE, bg=self.back)
        self.consoleFrame.grid(row=1, column=4, padx=5)
        yDefilB = tk.Scrollbar(self.root, orient='vertical')
        yDefilB.grid(row=1, column=5, sticky='ns')
        self.consoleTK = tk.Listbox(self.consoleFrame, font=self.font, yscrollcommand=yDefilB.set,
                                    height=self.hauteurs[0])
        self.consoleTK.grid(row=1, column=1, sticky='ns', pady=5)
        yDefilB['command'] = self.consoleTK.yview

        # Close window
        self.closeFrame = tk.LabelFrame(self.root, text="Close", relief=tk.RIDGE, bg=self.back)
        self.closeFrame.grid(row=2, column=4)
        self.close_button = tk.Button(self.closeFrame, text='Close', bg=self.back, font=self.font,
                                      command=self.closeWindow)
        self.close_button.grid(row=1, column=1, padx=10, pady=10)

        # Lancement
        self.root.mainloop()

    def log(self, texte):
        self.consoleTK.insert(tk.END, texte)
        print(texte)

    def getName(self):
        return self.networkname.get()

    def getLoader(self):
        print(self.loaderTK.get())
        return eval(self.loaderTK.get())

    def getCost(self):
        convert = {'Quadratique': QuadraticCost, 'Cross-entropy': CrossEntropyCost}
        userInput = self.cost_func.get()
        return convert[userInput]

    def getNetwork(self):
        tab = self.networkTK.get("1.0", tk.END)
        tab = tab.replace(' ', '').split('\n')
        return [eval(elem) for elem in tab if elem != '']

    def getEpochs(self):
        return int(self.Epochs.get())

    def getMiniSize(self):
        return int(self.mini_size.get())

    def getEta(self):
        return float(self.eta.get())

    def getMonitoring(self):
        return self.train_acc, self.eval_acc

    def updateImage(self, image):
        self.image = image
        self.imgDisplayed = ImageTk.PhotoImage(image=self.image.resize((224, 224)))
        self.imageTK.configure(image=self.imgDisplayed)

    def getImage(self):
        filetypes = (
            ('image files', '*.ppm'),
        )
        path = fd.askopenfilename(filetypes=filetypes)
        self.updateImage(Image.open(path))

    def loadRandom(self):
        if self.data is None:
            self.data = self.loader()
            self.data = [list(x) for x in self.data]
        tableau = random.choice(self.data[0])[0]
        image = Image.fromarray(tableau)
        self.updateImage(image)

    def feedforward(self):
        print('ff...')
        ff = self.net.feedforwardClasse(np.asarray(self.image))
        self.log(f'Probable: {ff}')
        self.output['text'] = f'Prévision: \n{ff}'

    def create(self):
        self.net = network.Network(layers=self.getNetwork(), cost=self.getCost())
        self.log('Network created')

    def save(self):
        name = self.getName()
        self.net.save(name)

    def load(self):
        name = self.getName()
        self.net = network.load(name)
        self.networkTK.delete("0.0", tk.END)

        for layer in self.net.layers:
            self.networkTK.insert(tk.END, '\n' + layer.name())

    def train(self):
        train_retour = self.net.train(loader=self.getLoader(), epochs=self.getEpochs(),
                                      mini_batch_size=self.getMiniSize(), eta=self.getEta(),
                                      train_accuracy=self.getMonitoring()[0], test_accuracy=self.getMonitoring()[1])
        return train_retour

    def eval_train(self):
        test_data, train_data = self.loader()
        self.net.accuracy(train_data, '(test)')

    def eval_test(self):
        test_data, train_data = self.loader()
        self.net.accuracy(test_data, '(test)')

    def closeWindow(self):
        self.root.destroy()


# test = Software(network=network.Network(
#    layers=[Maxpooling((64, 64, 3), 16), Convolution((4, 4, 3), 3, 5, sigmoid), Reshape((2, 2, 5), (2*2*5, 1)), Dense(20, 164, sigmoid)]),
#    loader=pano_loader)

# test = Software(network=network.Network(layers=[Maxpooling((64, 64, 3), 2), Convolution((32, 32, 3), 4, 5, sigmoid), Reshape((29, 29, 5), (5 * 29 * 29, 1)),
#            Dense(29*29* 5, 512, sigmoid), Dense(512, 164, sigmoid)]), loader=pano_loader_small)


# test = Software(
#   network=network.Network(layers=[Convolution((64, 64, 3), 4, 5, sigmoid), Reshape((61, 61, 5), (5 * 61 * 61, 1)),
#                                  Dense(61 * 61 * 5, 164, sigmoid)]),
# loader=pano_loader_medium)

test = Software(
    network=network.Network(layers=[Convolution((32, 32, 3), 4, 5, sigmoid), Reshape((29, 29, 5), (29 * 29 * 5, 1)),
                                    Dense(29 * 29 * 5, 164, sigmoid)]),
    loader=pano_loader_32)
