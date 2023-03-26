import pickle

def pano_loader_hidden(x):
    print('loading data')
    with open(f'./data/{x}', 'rb') as f:
        data = pickle.load(f)
    print('data loaded')
    return data['train'], data['test']

def pano_loader_grey():
    """
    :return: (train: [num, haut, larg, prof, y], test: [num, haut, larg, prof, y])
    test et train sont des numpy array de shape (nb_images, 32, 32, 1, 1)
    """
    return pano_loader_hidden('grey_255_LHE.pickle')

def pano_loader():
    """
    :return: (train: [num, haut, larg, prof, y], test: [num, haut, larg, prof, y])
    test et train sont des numpy array de shape (nb_images, 32, 32, 3, 1)
    """
    return pano_loader_hidden('255_mean.pickle')
