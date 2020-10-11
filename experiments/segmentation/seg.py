# System libs
import os, csv, torch, numpy, scipy.io, PIL.Image, torchvision.transforms
# Our libs
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode

colors = scipy.io.loadmat('../../data/color150.mat')['colors']
names = {}

with open('../../data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]

def visualize_result(img, pred, index=None):
    # filter prediction class if requested
    if index is not None:
        pred = pred.copy()
        pred[pred != index] = -1
        print(f'{names[index+1]}:')
        
    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(numpy.uint8)

    # aggregate images and save
    im_vis = numpy.concatenate((img, pred_color), axis=1)
    PIL.Image.fromarray(im_vis).save('result.png')

# Network Builders

# upernet resnet101
# net_encoder = ModelBuilder.build_encoder(
#     arch='resnet101',
#     fc_dim=2048,
#     weights='ckpt/ade20k-resnet101-upernet/encoder_epoch_50.pth')
# net_decoder = ModelBuilder.build_decoder(
#     arch='upernet',
#     fc_dim=2048,
#     num_class=150,
#     weights='ckpt/ade20k-resnet101-upernet/decoder_epoch_50.pth',
#     use_softmax=True)
# hrnet
net_encoder = ModelBuilder.build_encoder(
    arch='hrnetv2',
    fc_dim=720,
    weights='../../ckpt/ade20k-hrnetv2-c1/encoder_epoch_30.pth')
net_decoder = ModelBuilder.build_decoder(
    arch='c1',
    fc_dim=720,
    num_class=150,
    weights='../../ckpt/ade20k-hrnetv2-c1/decoder_epoch_30.pth',
    use_softmax=True)

crit = torch.nn.NLLLoss(ignore_index=-1)
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
segmentation_module.eval()
segmentation_module.cuda()

pil_to_tensor = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
        std=[0.229, 0.224, 0.225])  # across a large photo dataset.
])
pil_image = PIL.Image.open('sample.png').convert('RGB')
img_original = numpy.array(pil_image)
img_data = pil_to_tensor(pil_image)
singleton_batch = {'img_data': img_data[None].cuda()}
output_size = img_data.shape[1:]

with torch.no_grad():
    scores = segmentation_module(singleton_batch, segSize=output_size)
    
# Get the predicted scores for each pixel
_, pred = torch.max(scores, dim=1)
pred = pred.cpu()[0].numpy()

# Mask All
visualize_result(img_original, pred)

# Mask Human 
mask_idx = (pred==12) # 12 is human class label
img_original[mask_idx] = 0
PIL.Image.fromarray(img_original).save('mask_human.png')

# Top classes in answer
# predicted_classes = numpy.bincount(pred.flatten()).argsort()[::-1]
# for c in predicted_classes[:15]:
#     visualize_result(img_original, pred, c)