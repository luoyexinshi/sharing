import argparse

__author__ = 'Luoyexin Shi'

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--image_path', type=str, default=False, help='Pass in a single image', required = True) ## how to access it??
parser.add_argument('--checkpoint', type=str, default=False, help='checkpoint path', required=True)
parser.add_argument('--topk', type=int, default=5, help='top k probabilities')
parser.add_argument('--cat_to_name', type=str, default='cat_to_name.json', help='path to category to flower name mapping json')
parser.add_argument('--gpu', type=bool, default=False, help='whether to use gpu') #
args = parser.parse_args()

image_path = args.image_path
checkpoint = args.checkpoint
topk = args.topk
cat_to_name = args.cat_to_name
gpu = args.gpu


## Loading the checkpoint and rebuild the models# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_checkpoint(filepath):
    model = getattr(models, checkpoint['arch_name'])(pretrained=True)
    #set gradients to zero as done during training
    optimizer.zero_grad()
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_idx']
    model.epoch = checkpoint['epoch']
    model.optimizer = checkpoint['optimizer']

    return model

model = load_checkpoint('checkpoint.pth')
print(model)

# Preprossing Image
from PIL import Image

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    size = 256, 256
    image = Image.open(image)
    image.thumbnail(size, Image.ANTIALIAS)
    image = image.crop((
        size[0]//2 - 112,
        size[1]//2 - 112,
        size[0]//2 + 112,
        size[1]//2 + 112)
    )
    np_image = np.array(image)
    #Scale Image per channel
    # Using (image-min)/(max-min)
    np_image = np_image/255

    #normalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image- mean)/std

    np_image = np.transpose(np_image, (2,0,1))

    return np_image

## Class Prediction
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    model.to('cuda')
    image = process_image(image_path)
    image = torch.cuda.FloatTensor([image])

    model.eval()
    output = model.forward(image)
    ps = torch.exp(output).cpu().data.numpy()[0]

    topk_index = np.argsort(ps)[-topk:][::-1]
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    topk_class = [idx_to_class[x] for x in topk_index]
    topk_prob = ps[topk_index]

    return topk_prob, topk_class

prob,classes = predict(image_path, model)
