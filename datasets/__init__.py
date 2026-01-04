import torch
import torchvision.transforms as transforms
from .basedataset import namedataset,labeldataset
from PIL import Image

def get_dataset(phase, image_size, data_path, grayscale=False):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5) if not grayscale else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose(
        [transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC), transforms.ToTensor()]
    )
    if phase == 'train':
        dataset = labeldataset(
            data_path,
            transform = train_transform,
            grayscale = grayscale,
        )
    else:
        dataset = namedataset(
            data_path,
            transform=test_transform,
            grayscale = grayscale,
        )
    return dataset

def rescale(X):
    X = 2 * X - 1.0
    return X

def inverse_rescale(X):
    X = (X + 1.0) / 2.0
    return torch.clamp(X, 0.0, 1.0)

def imageresize2tensor(path, image_size, grayscale=False):
    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    convert = transforms.Compose(
        [transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC), transforms.ToTensor()]
    )
    return convert(img)

def image2tensor(path, grayscale=False):
    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    convert_tensor = transforms.ToTensor()
    return convert_tensor(img)
