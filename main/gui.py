import random

import network
from loaders.mnist_loader import *
from loaders.pano_loader import *
from activation_functions import *
from layers import *
from cost_functions import *

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog as fd
from PIL import ImageTk, Image


class Software(object):

    def __init__(self, network, loader, background='#FFEFDB', size='1000x600', windows_title='CNN - Dupuis Benjamin'):

        self.loader = loader
        self.net = network
        self.loader = loader

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

        # Settings frame
        self.settings = tk.LabelFrame(self.root, text="Paramètres entrainement", relief=tk.RIDGE, bg=self.back)
        self.settings.grid(row=1, column=2)

        tk.Label(self.settings, text="Nombre epochs", font=self.font, bg=self.back).grid(row=1, column=1,
                                                                                         padx=10, pady=10)
        self.Epochs = tk.Entry(self.settings)
        self.Epochs.insert(0, '1')
        self.Epochs.grid(row=1, column=2, padx=10, pady=10)

        tk.Label(self.settings, text="Taille mini-batch", font=self.font, bg=self.back).grid(row=2, column=1,
                                                                                             padx=10, pady=10)
        self.mini_size = tk.Entry(self.settings)
        self.mini_size.insert(0, '15')
        self.mini_size.grid(row=2, column=2, padx=10, pady=10)

        tk.Label(self.settings, text="Taux d'apprentissage", font=self.font, bg=self.back).grid(row=3,
                                                                                                column=1,
                                                                                                padx=10,
                                                                                                pady=10)
        self.eta = tk.Entry(self.settings)
        self.eta.insert(0, '3.0')
        self.eta.grid(row=3, column=2, padx=10, pady=10)

        # Resultat frame
        self.resultat = tk.LabelFrame(self.root, text="Entrainement/Evaluation", relief=tk.RIDGE, bg=self.back)
        self.resultat.grid(row=2, column=2)

        tk.Label(self.resultat, text="Quel resultats (graph)?\n texte pour l'instant", font=self.font,
                 bg=self.back).grid(row=1, column=1, padx=10, pady=10)

        self.eval_acc, self.train_acc = tk.BooleanVar(), tk.BooleanVar()

        self.eval_acc_button = tk.Checkbutton(self.resultat, text='Evaluation Accuracy', bg=self.back,
                                              variable=self.eval_acc)
        self.eval_acc_button.grid(row=2, column=2, padx=10, pady=10)

        self.train_acc_button = tk.Checkbutton(self.resultat, text='Training Accuracy', bg=self.back,
                                               variable=self.train_acc)
        self.train_acc_button.grid(row=4, column=2, padx=10, pady=10)

        self.train_button = tk.Button(self.resultat, text='Train Network', bg=self.back, font=self.font,
                                      command=self.train)
        self.train_button.grid(row=2, column=1, padx=10, pady=10)

        self.eval_button = tk.Button(self.resultat, text='Eval Network', bg=self.back, font=self.font,
                                     command=self.eval)
        self.eval_button.grid(row=4, column=1, padx=10, pady=10)

        # Image frame
        self.imageFrame = tk.LabelFrame(self.root, text="Image", relief=tk.RIDGE, bg=self.back)
        self.imageFrame.grid(row=1, column=3)

        self.image = Image.open("C:/Users/benyo/Desktop/Images/fond.png").resize((224, 224), resample=0)
        self.image = self.image.convert('RGB')
        self.imgDisplayed = ImageTk.PhotoImage(image=self.image)
        self.Image = tk.Label(self.imageFrame, image=self.imgDisplayed)
        self.Image.grid(row=1, column=1)

        self.outputFrame = tk.LabelFrame(self.root, text="Output", relief=tk.RIDGE, bg=self.back)
        self.outputFrame.grid(row=2, column=3)
        self.output = tk.Label(self.outputFrame, text='None')
        self.output.grid(row=1, column=1)

        self.eval_button = tk.Button(self.imageFrame, text='Feedforward Image', bg=self.back, font=self.font,
                                     command=self.feedforward)
        self.eval_button.grid(row=4, column=1, padx=10, pady=10)

        self.loadImage = tk.Button(self.imageFrame, text='Load Image', bg=self.back, font=self.font,
                                   command=self.getImage)
        self.loadImage.grid(row=3, column=1, padx=10, pady=10)

        self.randomImage = tk.Button(self.imageFrame, text='Load random', bg=self.back, font=self.font,
                                   command=self.loadRandom)
        self.randomImage.grid(row=2, column=1, padx=10, pady=10)

        # Load/Save frame
        self.network = tk.LabelFrame(self.root, text="Choose Network", relief=tk.RIDGE, bg=self.back)
        self.network.grid(row=1, column=1)

        tk.Label(self.network, text="Network Name", font=self.font, bg=self.back).grid(row=1, column=1, padx=10,
                                                                                       pady=10)
        self.networkname = tk.Entry(self.network)
        self.networkname.insert(0, 'test')
        self.networkname.grid(row=1, column=2, padx=10, pady=10)

        tk.Label(self.network, text="Fonction Cout", font=self.font, bg=self.back).grid(row=2, column=1,
                                                                                        padx=10, pady=10)
        self.cost_func = ttk.Combobox(self.network, values=['Quadratique', 'Cross-entropy'])
        self.cost_func.insert(0, 'Quadratique')
        self.cost_func.grid(row=2, column=2, padx=10, pady=10)

        tk.Label(self.network, text="Data Loader", font=self.font, bg=self.back).grid(row=3, column=1, padx=10,
                                                                                      pady=10)
        self.loaderTK = tk.Entry(self.network)
        self.loaderTK.insert(0, loader.__name__)
        self.loaderTK.grid(row=3, column=2, padx=10, pady=10)

        self.save = tk.Button(self.network, text='Save as name', bg=self.back, font=self.font, command=self.save)
        self.save.grid(row=4, column=1, padx=10, pady=10)

        self.load = tk.Button(self.network, text='Load by name', bg=self.back, font=self.font, command=self.load)
        self.load.grid(row=4, column=2, padx=10, pady=10)

        self.networkTK = tk.Text(self.network, bg='white', font=self.font, width=30, height=10)
        self.networkTK.grid(row=5, column=2, padx=10, pady=10)

        self.createTK = tk.Button(self.network, text='Create', bg=self.back, font=self.font, command=self.create)
        self.createTK.grid(row=5, column=1, padx=10, pady=10)

        # Instructions
        self.text = tk.LabelFrame(self.root, text="Instructions", relief=tk.RIDGE, bg=self.back)
        self.text.grid(row=2, column=1)
        tk.Label(self.text, text="Couches:\n\n"
                                 "Convolution, Dense, Reshape, Softmax\n\n"
                                 "", font=self.font, bg=self.back).grid(row=4, column=1,
                                                                        pady=10, padx=10)

        # Lancement
        self.root.mainloop()

    def getName(self): return self.networkname.get()
    def getLoader(self): return eval(self.loaderTK.get())
    def getCost(self):
        convert = {'Quadratique': QuadraticCost, 'Cross-entropy': CrossEntropyCost}
        userInput = self.cost_func.get()
        return convert[userInput]
    def getNetwork(self):
        tab = self.networkTK.get("1.0", tk.END)
        tab = tab.replace(' ', '').split('\n')
        print(type(tab))
        print(tab)
        return [eval(elem) for elem in tab if elem != '']

    def getEpochs(self): return int(self.Epochs.get())
    def getMiniSize(self): return int(self.mini_size.get())
    def getEta(self): return float(self.eta.get())

    def getMonitoring(self): return self.eval_acc, self.train_acc

    def updateImage(self, image):
        self.image = image
        self.imgDisplayed = ImageTk.PhotoImage(image=self.image.resize((224, 224)))
        self.Image.configure(image=self.imgDisplayed)

    def getImage(self):
        filetypes = (
            ('image files', '*.png'),
        )
        path = fd.askopenfilename(filetypes=filetypes)
        self.updateImage(Image.open(path))

    def loadRandom(self):
        train, test = self.loader()
        tableau = random.choice(list(test))[0]
        print(type(tableau))
        print(tableau.shape)
        image = Image.fromarray(tableau)
        self.updateImage(image)

    def feedforward(self):
        print(self.image.convert('RGB').mode)
        ff = self.net.feedforward(np.moveaxis(np.asarray(self.image), 0, -1))
        self.output['text'] = f'{ff} \n \nChiffre Probable: \n{np.argmax(ff)}'

    def create(self):
        self.net = network3.Network(layers=self.getNetwork(), cost=self.getCost())
        print('network initialized')

    def save(self):
        name = self.getName()
        self.net.save(name)

    def load(self):
        name = self.getName()
        try:
            self.net = network3.load(name)
            self.net_entry.delete(0, tk.END)
            self.net_entry.insert(0, self.net.sizes)

            convert = {QuadraticCost: 'Quadratique', CrossEntropyCost: 'Cross-Entropy'}
            self.cost_func.delete(0, tk.END)
            self.cost_func.insert(0, convert[self.net.cost])

            convert_func = {sigmoid: 'Sigmoid', tanh: 'Tanh', identity: 'Identity'}
            self.activ_func.delete(0, tk.END)
            self.activ_func.insert(0, convert_func[self.net.activation])
        except:
            print('Error while loading')

    def train(self):
        self.train_retour = self.net.train(loader=self.getLoader())

    def eval(self):
        test_data, train_data = self.loader()
        self.net.accuracy(test_data, '(test)')

    def CloseWindow(self):
        self.root.destroy()


test = Software(network=network3.Network(layers=[Reshape((64,64,3),(64*64*3,1)), Dense(64*64*3, 256, sigmoid), Dense(256, 164, sigmoid)]),
                loader=pano_loader)
