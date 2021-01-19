import torchvision.transforms as transforms

new_size = (400,400)
data_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize(new_size)])

def preprocessing(img):
    # Resize the image and convert it to tensors
    img = data_transforms(img)

    # Change the image shape from [W, H, C] to [B, C, W, H]
    # Where B = Batch Size, C = Channels (RGB), W = Width, H = Height.
    img = img.unsqueeze(0)

    return img
