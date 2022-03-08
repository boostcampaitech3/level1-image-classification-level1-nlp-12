import os
import pandas as pd
from PIL import Image
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize,CenterCrop,ColorJitter
from dataset import TestDataset, MaskBaseDataset


class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
 
    submission = pd.read_csv(os.path.join(data_dir, 'info.csv'))
    image_dir = os.path.join(data_dir, 'images')

    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    transform = transforms.Compose([
        CenterCrop((370,320)),
        Resize((272,232), Image.BILINEAR),
       # ColorJitter(0.2, 0.2, 0.2, 0.2),
        ToTensor(),
        Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)),
    ])
    dataset = TestDataset(image_paths, transform)

    loader = DataLoader(
        dataset,
        shuffle=False,
        num_workers=4,
        batch_size=32,
        #pin_memory=use_cuda,
        drop_last=False,

    )

    # 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
    device = torch.device('cuda')
    #model = NfnetModel().to(device=device)
    #model.load_state_dict(copyStateDict(torch.load('../models/best.pth')["state_dict"]))
    #model.eval()
    model = torch.load( os.path.join(args.model_dir, args.model_name)).to(device)
    model.eval()

    # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
    all_predictions = []
    for images in loader:
        with torch.no_grad():
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())
    submission['ans'] = all_predictions
    submission_name = '_'.join([args.model_name ,'submission.csv'])
    # 제출할 파일을 저장합니다.
    submission.to_csv(os.path.join(output_dir, submission_name), index=False)
    print('test inference is done!')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for validing (default: 32)')
    parser.add_argument('--resize', type=tuple, default=(96, 128), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './model'))
    
    # 필수 
    parser.add_argument('--model_name', type=str, default='None', help='model name in model directory (default: None)')

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)