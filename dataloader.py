import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets

class dSprites(Dataset):
    '''
    load from npz files

    '''
    def __init__(self, npz_path, transform=None, rtn_latent=False):
        super().__init__()
        npz = np.load(npz_path)
        self.transform = transform
        self.rtn_latent = rtn_latent
        self.imgs = npz['imgs'] # N * 64 * 64, uint8
        self.num = 737280
        assert self.num == self.imgs.shape[0]
        if rtn_latent:
            self.latents = npz['latents_values'] # N * 6, float
            self.classes = npz['latents_classes'] # N * 6, int

    def __getitem__(self, index):
        img = self.imgs[index] * 255
        img = Image.fromarray(img) # 64 * 64
        if self.transform is not None:
            img = self.transform(img)
        if self.rtn_latent:
            latent = self.latents[index]
            class_ = self.classes[index]
            return img, latent, class_
        else:
            return img

    def __len__(self):
        return self.num

# Directory containing the data.
root = 'data/'

def get_data(dataset, batch_size):

    # Get MNIST dataset.
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor()])

        dataset = dsets.MNIST(root+'mnist/', train='train', 
                                download=True, transform=transform)

    # Get SVHN dataset.
    elif dataset == 'SVHN':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor()])

        dataset = dsets.SVHN(root+'svhn/', split='train', 
                                download=True, transform=transform)

    # Get FashionMNIST dataset.
    elif dataset == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor()])

        dataset = dsets.FashionMNIST(root+'fashionmnist/', train='train', 
                                download=True, transform=transform)

    # Get CelebA dataset.
    # MUST ALREADY BE DOWNLOADED IN THE APPROPRIATE DIRECTOR DEFINED BY ROOT PATH!
    elif dataset == 'CelebA':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5))])

        dataset = dsets.ImageFolder(root=root+'celeba/', transform=transform)

    # Get dSprites dataset.
    elif dataset == 'dSprites':
        transform = transforms.ToTensor()

        dataset = dSprites(root+'dsprites/dsprites.npz', transform=transform)
    
    # Create dataloader.
    dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

    return dataloader