##  :penguin: 자, 연어 한접시's First P-Stage Project :penguin:
부스트 캠프 AI Tech 3기 12조의 마스크 착용 상태 분류 프로젝트

## :mag_right: Overview
### Background
> COVID-19의 확산으로 우리나라는 물론 전 세계 사람들은 경제적, 생산적인 활동에 제약을 받게 되었다. 
> 이번 프로젝트는 사진 속 사람의 특징에 따라 분류를 해야하는 Image Classification Task에 속한다
> 단순하게 마스크 착용 여부만 판단하는 것이 아니라, 잘못 착용한 경우(코스크, 턱스크 등등) 도 같이 분류해야 한다.
> 추가적으로 성별과 연령대(30대 이하, 30~60, 60대 이상)도 올바르게 분류할 수 있어야 한다.

### Problem definition
> (384, 512) 사람 이미지가 주어지면 나이, 성별, 마스크 착용 여부에 따라 18개의 클래스로 Classification 하는 모델 구현
![img](./material/class.png)

### Development Environment
    개발환경(Hardware) : aistage에서 제공하는 서버 및 GPU(Tesla V100)
    개발환경(IDE) : Jupyter notebook, VSCode, PyCharm 등
    협업 및 기타 Tool : Github, Notion, Zoom, TensorBoard, wandb

#### Project Tree
```bash
level1-image-classification-level1-nlp-12
├── EDA
│   ├── 001_Data_Analysis.ipynb
│   ├── 002_Image_Blend.ipynb
│   ├── 003_wideResnet50Model.ipynb
│   ├── EDA_001.ipynb
│   └── README.md
├── README.md
├── dataset.py
├── inference.py
├── loss.py
├── model.py
├── project_tree.txt
├── requirments.txt
└── train.py
```

## :page_facing_up: Wrap Up Report
[Wrap Up Report](./material/WrapUp%20%EB%A6%AC%ED%8F%AC%ED%8A%B8.pdf)