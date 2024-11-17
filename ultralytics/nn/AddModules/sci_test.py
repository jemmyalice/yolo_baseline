import os
import sys
import numpy as np
import torch
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from PIL import Image
from SCINet import Finetunemodel
from multi_read_data import MemoryFriendlyLoader

# Argument parser
parser = argparse.ArgumentParser("SCI")
parser.add_argument('--data_path', type=str, default='./data/medium',
                    help='location of the data corpus')
parser.add_argument('--save_path', type=str, default='./results/medium', help='location to save results')
parser.add_argument('--model', type=str, default='./weights/medium.pt', help='location of the model weights')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')

args = parser.parse_args()
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)

# Load dataset
TestDataset = MemoryFriendlyLoader(img_dir=args.data_path, task='test')
test_queue = torch.utils.data.DataLoader(
    TestDataset, batch_size=1,
    pin_memory=True, num_workers=0)

# Function to save images
def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')
    print(f"Saving results to {path}")


def main():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print('No GPU device available, using CPU.')

    # Load the model
    model = Finetunemodel(args.model)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True), strict=False)
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        for _, (input, image_name) in enumerate(test_queue):
            input = input.to(device)  # Ensure input is on the correct device
            image_name = image_name[0].split('\\')[-1].split('.')[0]
            i, r = model(input)
            u_name = f'{image_name}1.png'
            print(f'Processing {u_name}')
            u_path = os.path.join(save_path, u_name)
            save_images(r, u_path)
            print('we have save it')

if __name__ == '__main__':
    main()
