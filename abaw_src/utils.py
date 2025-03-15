
import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from torch import distributed as dist
import random
import numpy as np
from PIL import Image

class Clipdataset(Dataset):
    def __init__(self, train_csv_path, root_dir='/workspace/ABAW8/', transform=None, sequence_length=32, seed=42):
        """
        Args:
            train_csv_path (str): train.csv 파일 경로
            root_dir (str): 이미지들이 저장된 기본 경로
            transform (callable, optional): 데이터 전처리 함수
            sequence_length (int): 반환할 이미지와 레이블 수
            seed (int): 랜덤 시드 값 (재현성을 위해 사용)
        """
        self.train_df = pd.read_csv(train_csv_path)
        self.root_dir = root_dir
        self.sequence_length = sequence_length

        # 랜덤 시드 고정
        self.set_seed(seed)

        # 기본 이미지 변환: 224x224로 리사이즈
        self.transform = transform if transform else transforms.Compose([ 
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        # 현재 데이터의 batch와 label 경로
        batch = self.train_df.iloc[idx]['batch']
        label_csv_path = self.train_df.iloc[idx]['path']

        # 이미지 폴더 경로: '/workspace/ABAW/cropped_data' + batch + 폴더명
        img_folder = os.path.join(self.root_dir, batch, label_csv_path.split('.')[0].split('/')[-1])

        # 레이블 불러오기
        labels_df = pd.read_csv(label_csv_path)
        labels = labels_df['label'].tolist()

        # frame_num에 해당하는 이미지 파일 이름을 가져오기
        frame_nums = labels_df['frame_num'].tolist()  # frame_num을 리스트로 가져옴

        # 이미지 수와 레이블 수 중 작은 값으로 제한
        total_samples = min(len(frame_nums), len(labels))

        # sequence_length보다 작은 경우 마지막 프레임과 라벨로 패딩
        if total_samples < self.sequence_length:
            pad_count = self.sequence_length - total_samples
            last_frame = frame_nums[-1]
            last_label = labels[-1]

            # 마지막 프레임과 라벨을 추가
            frame_nums += [last_frame] * pad_count
            labels += [last_label] * pad_count
            total_samples = self.sequence_length

        # 무작위로 시작 지점 선택
        start_idx = random.randint(0, total_samples - self.sequence_length)

        # 이미지와 레이블 추출
        images, flipped_images, targets= [], [], []

        for i in range(start_idx, start_idx + self.sequence_length):
            frame_num = frame_nums[i]  # frame_num을 가져옴

            frame_num_str = f"{int(frame_num.split('.')[0]):05d}"
            img_path = os.path.join(img_folder, f"{frame_num_str}.jpg")
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = self.transform(img)
            flipped_image = transforms.RandomHorizontalFlip(p=1)(img)

            images.append(img)
            flipped_images.append(flipped_image)
            targets.append(labels[i])


        images = torch.stack(images)  # (sequence_length, C, 224, 224)
        flipped_images = torch.stack(flipped_images)
        targets = torch.tensor(targets)  # (sequence_length,)

        return images, flipped_images, targets




class Clipdataset_va(Dataset):
    def __init__(self, train_csv_path, root_dir='/workspace/ABAW8/', transform=None, sequence_length=32, seed=42):
        """
        Args:
            train_csv_path (str): train.csv 파일 경로
            root_dir (str): 이미지들이 저장된 기본 경로
            transform (callable, optional): 데이터 전처리 함수
            sequence_length (int): 반환할 이미지와 레이블 수
            seed (int): 랜덤 시드 값 (재현성을 위해 사용)
        """
        self.train_df = pd.read_csv(train_csv_path)
        self.root_dir = root_dir
        self.sequence_length = sequence_length

        # 랜덤 시드 고정
        self.set_seed(seed)

        # 기본 이미지 변환: 224x224로 리사이즈
        self.transform = transform if transform else transforms.Compose([ 
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        # 현재 데이터의 batch와 label 경로
        batch = self.train_df.iloc[idx]['batch']
        label_csv_path = self.train_df.iloc[idx]['path']

        # 이미지 폴더 경로: '/workspace/ABAW/cropped_data' + batch + 폴더명
        img_folder = os.path.join(self.root_dir, batch, label_csv_path.split('.')[0].split('/')[-1])

        # 레이블 불러오기
        labels_df = pd.read_csv(label_csv_path)
        vals = labels_df['val'].tolist()
        aros = labels_df['aro'].tolist()

        # frame_num에 해당하는 이미지 파일 이름을 가져오기
        frame_nums = labels_df['frame_num'].tolist()  # frame_num을 리스트로 가져옴

        # 이미지 수와 레이블 수 중 작은 값으로 제한
        total_samples = min(len(frame_nums), len(vals))

        # sequence_length보다 작은 경우 마지막 프레임과 라벨로 패딩
        if total_samples < self.sequence_length:
            pad_count = self.sequence_length - total_samples
            last_frame = frame_nums[-1]
            last_vals = vals[-1]
            last_aros = aros[-1]

            # 마지막 프레임과 라벨을 추가
            frame_nums += [last_frame] * pad_count
            vals += [last_vals] * pad_count
            aros += [last_aros] * pad_count
            total_samples = self.sequence_length

        # 무작위로 시작 지점 선택
        start_idx = random.randint(0, total_samples - self.sequence_length)

        # 이미지와 레이블 추출
        images, flipped_images, targets= [], [], []

        for i in range(start_idx, start_idx + self.sequence_length):
            frame_num = frame_nums[i]  # frame_num을 가져옴
            frame_num_str = f"{int(frame_num.split('.')[0]):05d}"
            img_path = os.path.join(img_folder, f"{frame_num_str}.jpg")
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = self.transform(img)
            flipped_image = transforms.RandomHorizontalFlip(p=1)(img)

            images.append(img)
            flipped_images.append(flipped_image)
            targets.append([vals[i], aros[i]])


        images = torch.stack(images)  # (sequence_length, C, 224, 224)
        flipped_images = torch.stack(flipped_images)
        targets = torch.tensor(targets)  # (sequence_length,)

        return images, flipped_images, targets

class Clipdataset_au(Dataset):
    def __init__(self, train_csv_path, root_dir='/workspace/ABAW8/', transform=None, sequence_length=32, seed=42):
        """
        Args:
            train_csv_path (str): train.csv 파일 경로
            root_dir (str): 이미지들이 저장된 기본 경로
            transform (callable, optional): 데이터 전처리 함수
            sequence_length (int): 반환할 이미지와 레이블 수
            seed (int): 랜덤 시드 값 (재현성을 위해 사용)
        """
        self.train_df = pd.read_csv(train_csv_path)
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.au_columns = ["AU1", "AU2", "AU4", "AU6", "AU7", "AU10", 
              "AU12", "AU15", "AU23", "AU24", "AU25", "AU26"]

        # 랜덤 시드 고정
        self.set_seed(seed)

        # 기본 이미지 변환: 224x224로 리사이즈
        self.transform = transform if transform else transforms.Compose([ 
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        # 현재 데이터의 batch와 label 경로
        batch = self.train_df.iloc[idx]['batch']
        label_csv_path = self.train_df.iloc[idx]['path']

        # 이미지 폴더 경로: '/workspace/ABAW/cropped_data' + batch + 폴더명
        img_folder = os.path.join(self.root_dir, batch, label_csv_path.split('.')[0].split('/')[-1])
        # 레이블 불러오기
        labels_df = pd.read_csv(label_csv_path)
        au_lists = [labels_df[au].tolist() for au in self.au_columns]


        # frame_num에 해당하는 이미지 파일 이름을 가져오기
        frame_nums = labels_df['frame_num'].tolist()  # frame_num을 리스트로 가져옴

        # 이미지 수와 레이블 수 중 작은 값으로 제한
        total_samples = min(len(frame_nums), len(au_lists))

        # sequence_length보다 작은 경우 마지막 프레임과 라벨로 패딩
        if total_samples < self.sequence_length:
            pad_count = self.sequence_length - total_samples
            last_frame = frame_nums[-1]
            last_au = [au[-1] for au in au_lists]


            # 마지막 프레임과 라벨을 추가
            frame_nums += [last_frame] * pad_count
            # 각 AU 리스트에 동일한 `last_au` 값 패딩 추가
            au_lists = [au + [last_au[i]] * pad_count for i, au in enumerate(au_lists)]
            total_samples = self.sequence_length

        # 무작위로 시작 지점 선택
        start_idx = random.randint(0, total_samples - self.sequence_length)

        # 이미지와 레이블 추출
        images, flipped_images, targets= [], [], []

        for i in range(start_idx, start_idx + self.sequence_length):
            frame_num = frame_nums[i]  # frame_num을 가져옴
            frame_num_str = f"{int(frame_num.split('.')[0]):05d}"
            img_path = os.path.join(img_folder, f"{frame_num_str}.jpg")
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = self.transform(img)
            flipped_image = transforms.RandomHorizontalFlip(p=1)(img)

            images.append(img)
            flipped_images.append(flipped_image)
            target_values = [au[i] for au in au_lists]  # AU1~AU26 값을 리스트로 저장
            targets.append(target_values)  # targets 리스트에 추가
    

        images = torch.stack(images)  # (sequence_length, C, 224, 224)
        flipped_images = torch.stack(flipped_images)
        targets = torch.tensor(targets)  # (sequence_length,)

        return images, flipped_images, targets
    

class Clipdataset_valid_au(Dataset):
    def __init__(self, train_csv_path, root_dir='/workspace/ABAW8/', transform=None, sequence_length=32, seed=42):
        """
        Args:
            train_csv_path (str): train.csv 파일 경로
            root_dir (str): 이미지들이 저장된 기본 경로
            transform (callable, optional): 데이터 전처리 함수
            sequence_length (int): 반환할 이미지와 레이블 수
            seed (int): 랜덤 시드 값 (재현성을 위해 사용)
        """
        self.train_df = pd.read_csv(train_csv_path)
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.au_columns = ["AU1", "AU2", "AU4", "AU6", "AU7", "AU10", 
              "AU12", "AU15", "AU23", "AU24", "AU25", "AU26"]

        # 랜덤 시드 고정
        self.set_seed(seed)

        # 기본 이미지 변환: 224x224로 리사이즈
        self.transform = transform if transform else transforms.Compose([ 
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        # 현재 데이터의 batch와 label 경로
        batch = self.train_df.iloc[idx]['batch']
        label_csv_path = self.train_df.iloc[idx]['path']

        # 이미지 폴더 경로: '/workspace/ABAW/cropped_data' + batch + 폴더명
        img_folder = os.path.join(self.root_dir, batch, label_csv_path.split('.')[0].split('/')[-1])
        # 레이블 불러오기
        labels_df = pd.read_csv(label_csv_path)
        au_lists = [labels_df[au].tolist() for au in self.au_columns]


        # frame_num에 해당하는 이미지 파일 이름을 가져오기
        frame_nums = labels_df['frame_num'].tolist()  # frame_num을 리스트로 가져옴

        # 이미지 수와 레이블 수 중 작은 값으로 제한
        total_samples = min(len(frame_nums), len(au_lists))

        # sequence_length보다 작은 경우 마지막 프레임과 라벨로 패딩
        if total_samples < self.sequence_length:
            pad_count = self.sequence_length - total_samples
            last_frame = frame_nums[-1]
            last_au = [au[-1] for au in au_lists]


            # 마지막 프레임과 라벨을 추가
            frame_nums += [last_frame] * pad_count
            # 각 AU 리스트에 동일한 `last_au` 값 패딩 추가
            au_lists = [au + [last_au[i]] * pad_count for i, au in enumerate(au_lists)]
            total_samples = self.sequence_length

        # 무작위로 시작 지점 선택
        start_idx = random.randint(0, total_samples - self.sequence_length)

        # 이미지와 레이블 추출
        images, targets= [], []

        for i in range(start_idx, start_idx + self.sequence_length):
            frame_num = frame_nums[i]  # frame_num을 가져옴

            frame_num_str = f"{int(frame_num.split('.')[0]):05d}"
            img_path = os.path.join(img_folder, f"{frame_num_str}.jpg")
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = self.transform(img)

            images.append(img)
            target_values = [au[i] for au in au_lists]  # AU1~AU26 값을 리스트로 저장
            targets.append(target_values)  # targets 리스트에 추가


        images = torch.stack(images)  # (sequence_length, C, 224, 224)
        targets = torch.tensor(targets)  # (sequence_length,)

        return images, targets

    
class Clipdataset_erasing(Dataset):
    def __init__(self, train_csv_path, root_dir='/workspace/ABAW8/', transform=None, sequence_length=32, seed=42):
        """
        Args:
            train_csv_path (str): train.csv 파일 경로
            root_dir (str): 이미지들이 저장된 기본 경로
            transform (callable, optional): 데이터 전처리 함수
            sequence_length (int): 반환할 이미지와 레이블 수
            seed (int): 랜덤 시드 값 (재현성을 위해 사용)
        """
        self.train_df = pd.read_csv(train_csv_path)
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.aug_func = [flip_image, add_g]
        # 랜덤 시드 고정
        self.set_seed(seed)

        # 기본 이미지 변환: 224x224로 리사이즈
        self.transform = transform if transform else transforms.Compose([ 
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        # 현재 데이터의 batch와 label 경로
        batch = self.train_df.iloc[idx]['batch']
        label_csv_path = self.train_df.iloc[idx]['path']

        # 이미지 폴더 경로: '/workspace/ABAW/cropped_data' + batch + 폴더명
        img_folder = os.path.join(self.root_dir, batch, label_csv_path.split('.')[0].split('/')[-1])

        # 레이블 불러오기
        labels_df = pd.read_csv(label_csv_path)
        labels = labels_df['label'].tolist()

        # frame_num에 해당하는 이미지 파일 이름을 가져오기
        frame_nums = labels_df['frame_num'].tolist()  # frame_num을 리스트로 가져옴

        # 이미지 수와 레이블 수 중 작은 값으로 제한
        total_samples = min(len(frame_nums), len(labels))

        # sequence_length보다 작은 경우 마지막 프레임과 라벨로 패딩
        if total_samples < self.sequence_length:
            pad_count = self.sequence_length - total_samples
            last_frame = frame_nums[-1]
            last_label = labels[-1]

            # 마지막 프레임과 라벨을 추가
            frame_nums += [last_frame] * pad_count
            labels += [last_label] * pad_count
            total_samples = self.sequence_length

        # 무작위로 시작 지점 선택
        start_idx = random.randint(0, total_samples - self.sequence_length)

        # 이미지와 레이블 추출
        images, flipped_images, targets= [], [], []

        for i in range(start_idx, start_idx + self.sequence_length):
            frame_num = frame_nums[i]  # frame_num을 가져옴

            frame_num_str = f"{int(frame_num.split('.')[0]):05d}"
            img_path = os.path.join(img_folder, f"{frame_num_str}.jpg")
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            
            if random.uniform(0, 1) > 0.5:
                img = self.aug_func[1](img)
            img = self.transform(img)
            flipped_image = transforms.RandomHorizontalFlip(p=1)(img)
            


            images.append(img)
            flipped_images.append(flipped_image)
            targets.append(labels[i])


        images = torch.stack(images)  # (sequence_length, C, 224, 224)
        flipped_images = torch.stack(flipped_images)
        targets = torch.tensor(targets)  # (sequence_length,)

        return images, flipped_images, targets
    



class Clipdataset_concat(Dataset):
    def __init__(self, train_csv_path, root_dir='/workspace/ABAW8/', transform=None, sequence_length=32, seed=42):
        """
        Args:
            train_csv_path (str): train.csv 파일 경로
            root_dir (str): 이미지들이 저장된 기본 경로
            transform (callable, optional): 데이터 전처리 함수
            sequence_length (int): 반환할 이미지와 레이블 수
            seed (int): 랜덤 시드 값 (재현성을 위해 사용)
        """
        self.train_df = pd.read_csv(train_csv_path)
        self.root_dir = root_dir
        self.sequence_length = sequence_length

        # 랜덤 시드 고정
        self.set_seed(seed)

        # 기본 이미지 변환: 224x224로 리사이즈
        self.transform = transform if transform else transforms.Compose([ 
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        # 현재 데이터의 batch와 label 경로
        batch = self.train_df.iloc[idx]['batch']
        label_csv_path = self.train_df.iloc[idx]['path']

        # 이미지 폴더 경로: '/workspace/ABAW/cropped_data' + batch + 폴더명
        img_folder = os.path.join(self.root_dir, batch, label_csv_path.split('.')[0].split('/')[-1])

        # 레이블 불러오기
        labels_df = pd.read_csv(label_csv_path)
        labels = labels_df['label'].tolist()

        # frame_num에 해당하는 이미지 파일 이름을 가져오기
        frame_nums = labels_df['frame_num'].tolist()  # frame_num을 리스트로 가져옴

        # 이미지 수와 레이블 수 중 작은 값으로 제한
        total_samples = min(len(frame_nums), len(labels))

        # sequence_length보다 작은 경우 마지막 프레임과 라벨로 패딩
        if total_samples < self.sequence_length:
            pad_count = self.sequence_length - total_samples
            last_frame = frame_nums[-1]
            last_label = labels[-1]

            # 마지막 프레임과 라벨을 추가
            frame_nums += [last_frame] * pad_count
            labels += [last_label] * pad_count
            total_samples = self.sequence_length

        # 무작위로 시작 지점 선택
        start_idx = random.randint(0, total_samples - self.sequence_length)

        # 이미지와 레이블 추출
        images, flipped_images, targets= [], [], []

        for i in range(start_idx, start_idx + self.sequence_length):
            frame_num = frame_nums[i]  # frame_num을 가져옴

            frame_num_str = f"{int(frame_num.split('.')[0]):05d}"
            img_path = os.path.join(img_folder, f"{frame_num_str}.jpg")
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = self.transform(img)
            flipped_image = transforms.RandomHorizontalFlip(p=1)(img)

            images.append(img)
            flipped_images.append(flipped_image)
            combined_images = images + flipped_images
            targets.append(labels[i])


        combined_images = torch.stack(combined_images)  # (sequence_length, C, 224, 224)
        targets = torch.tensor(targets)  # (sequence_length,)

        return combined_images, targets




class Clipdataset_valid(Dataset):
    def __init__(self, train_csv_path, root_dir='/workspace/ABAW8/', transform=None, sequence_length=32):
        """
        Args:
            train_csv_path (str): train.csv 파일 경로
            root_dir (str): 이미지들이 저장된 기본 경로
            transform (callable, optional): 데이터 전처리 함수
            sequence_length (int): 반환할 이미지와 레이블 수
        """
        self.train_df = pd.read_csv(train_csv_path)
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        
        # 기본 이미지 변환: 224x224로 리사이즈
        self.transform = transform if transform else transforms.Compose([ 
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        # 현재 데이터의 batch와 label 경로
        batch = self.train_df.iloc[idx]['batch']
        label_csv_path = self.train_df.iloc[idx]['path']

        # 이미지 폴더 경로: '/workspace/ABAW/cropped_data' + batch + 폴더명
        img_folder = os.path.join(self.root_dir, batch, label_csv_path.split('.')[0].split('/')[-1])

        # 레이블 불러오기
        labels_df = pd.read_csv(label_csv_path)
        labels = labels_df['label'].tolist()

        # frame_num에 해당하는 이미지 파일 이름을 가져오기
        frame_nums = sorted(labels_df['frame_num'].tolist())  # frame_num을 리스트로 가져옴

        # 정렬
        sorted_indices = sorted(range(len(frame_nums)), key=lambda x: frame_nums[x])
        frame_nums = [frame_nums[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]

        # 이미지 수와 레이블 수 중 작은 값으로 제한
        total_samples = min(len(frame_nums), len(labels))

        # sequence_length보다 작은 경우 마지막 프레임과 라벨로 패딩
        if total_samples < self.sequence_length:
            pad_count = self.sequence_length - total_samples
            last_frame = frame_nums[-1]
            last_label = labels[-1]
            
            # 마지막 프레임과 라벨을 추가
            frame_nums += [last_frame] * pad_count
            labels += [last_label] * pad_count
            total_samples = self.sequence_length

        # 이미지와 레이블 추출
        images, targets, img_path = [], [], []

        for i in range(0, self.sequence_length):
            frame_num = frame_nums[i]  # frame_num을 가져옴
            # frame_num에서 .jpg를 제외하고 숫자로 변환한 후 5자리로 포맷
            frame_num_str = f"{int(frame_num.split('.')[0]):05d}"
            img_path = os.path.join(img_folder, f"{frame_num_str}.jpg")  # frame_num + '.jpg' 형식으로 이미지 경로 설정
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB 변환

            img = self.transform(img)  # 224x224로 리사이즈 및 텐서 변환

            images.append(img)
            targets.append(labels[i])

        images = torch.stack(images)  # (sequence_length, C, 224, 224)
        targets = torch.tensor(targets)  # (sequence_length,)

        return images, targets



class Clipdataset_valid_va(Dataset):
    def __init__(self, train_csv_path, root_dir='/workspace/ABAW8/', transform=None, sequence_length=32, seed=42):
        """
        Args:
            train_csv_path (str): train.csv 파일 경로
            root_dir (str): 이미지들이 저장된 기본 경로
            transform (callable, optional): 데이터 전처리 함수
            sequence_length (int): 반환할 이미지와 레이블 수
            seed (int): 랜덤 시드 값 (재현성을 위해 사용)
        """
        self.train_df = pd.read_csv(train_csv_path)
        self.root_dir = root_dir
        self.sequence_length = sequence_length

        # 랜덤 시드 고정
        self.set_seed(seed)

        # 기본 이미지 변환: 224x224로 리사이즈
        self.transform = transform if transform else transforms.Compose([ 
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        # 현재 데이터의 batch와 label 경로
        batch = self.train_df.iloc[idx]['batch']
        label_csv_path = self.train_df.iloc[idx]['path']

        # 이미지 폴더 경로: '/workspace/ABAW/cropped_data' + batch + 폴더명
        img_folder = os.path.join(self.root_dir, batch, label_csv_path.split('.')[0].split('/')[-1])

        # 레이블 불러오기
        labels_df = pd.read_csv(label_csv_path)
        vals = labels_df['val'].tolist()
        aros = labels_df['aro'].tolist()

        # frame_num에 해당하는 이미지 파일 이름을 가져오기
        frame_nums = labels_df['frame_num'].tolist()  # frame_num을 리스트로 가져옴

        # 이미지 수와 레이블 수 중 작은 값으로 제한
        total_samples = min(len(frame_nums), len(vals))

        # sequence_length보다 작은 경우 마지막 프레임과 라벨로 패딩
        if total_samples < self.sequence_length:
            pad_count = self.sequence_length - total_samples
            last_frame = frame_nums[-1]
            last_vals = vals[-1]
            last_aros = aros[-1]

            # 마지막 프레임과 라벨을 추가
            frame_nums += [last_frame] * pad_count
            vals += [last_vals] * pad_count
            aros += [last_aros] * pad_count
            total_samples = self.sequence_length

        # 무작위로 시작 지점 선택
        start_idx = random.randint(0, total_samples - self.sequence_length)

        # 이미지와 레이블 추출
        images, targets= [], []

        for i in range(start_idx, start_idx + self.sequence_length):
            frame_num = frame_nums[i]  # frame_num을 가져옴

            frame_num_str = f"{int(frame_num.split('.')[0]):05d}"
            img_path = os.path.join(img_folder, f"{frame_num_str}.jpg")
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = self.transform(img)

            images.append(img)
            targets.append([vals[i], aros[i]])


        images = torch.stack(images)  # (sequence_length, C, 224, 224)
        targets = torch.tensor(targets)  # (sequence_length,)

        return images, targets
    

def save_classifier(model, epoch, args):

    path = os.path.join(args.output_dir,f'{epoch}.pth')
    torch.save(model.state_dict(), path)
    print(f'save : {epoch} : {path}')

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

class AccuracyLogger_torch:
    """Computes and stores the average and current value, including F1 scores."""

    def __init__(self, num_class):
        self.num_class = num_class
        self.reset()

    def reset(self):
        self.classwise_sum = torch.zeros(self.num_class, dtype=torch.float32, device='cpu')
        self.classwise_count = torch.zeros(self.num_class, dtype=torch.float32, device='cpu')
        self.total_sum = torch.tensor(0.0, dtype=torch.float32, device='cpu')
        self.total_count = torch.tensor(0.0, dtype=torch.float32, device='cpu')

        self.true_positive = torch.zeros(self.num_class, dtype=torch.float32, device='cpu')
        self.false_positive = torch.zeros(self.num_class, dtype=torch.float32, device='cpu')
        self.false_negative = torch.zeros(self.num_class, dtype=torch.float32, device='cpu')
    
    def update(self, predictions, labels):
        predictions = predictions.flatten()
        labels = labels.flatten()

        # Update total values
        self.total_sum += (predictions == labels).sum().float()
        self.total_count += predictions.size(0)
        # Update class-wise values
        for i in range(self.num_class):
            class_mask = (labels == i) # label mask
            pred_mask = (predictions == i) # prediction mask

            self.classwise_sum[i] += (predictions[class_mask] == labels[class_mask]).sum().float()
            self.classwise_count[i] += class_mask.sum().float()

            # For F1 score calculation
            self.true_positive[i] += (pred_mask & class_mask).sum().float()
            self.false_positive[i] += (pred_mask & ~class_mask).sum().float()
            self.false_negative[i] += (~pred_mask & class_mask).sum().float()

    def gather(self):
        dist.barrier()
        dist.all_reduce(self.classwise_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.classwise_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.total_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.total_count, op=dist.ReduceOp.SUM)

        dist.all_reduce(self.true_positive, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.false_positive, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.false_negative, op=dist.ReduceOp.SUM)

    def final_score(self):
        # Calculate classwise accuracy
        classwise_acc = self.classwise_sum / (self.classwise_count + 1e-6)

        # Calculate total mean accuracy
        total_acc = self.total_sum / (self.total_count + 1e-6)

        # Calculate Precision, Recall, and F1 Score for each class
        precision = self.true_positive / (self.true_positive + self.false_positive + 1e-12)
        recall = self.true_positive / (self.true_positive + self.false_negative + 1e-12)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)

        # Replace NaN F1 scores (e.g., for classes with no true positives) with 0
        f1_scores = torch.nan_to_num(f1_scores, nan=0.0, posinf=0.0, neginf=0.0)

        # Calculate average F1 Score across all classes
        mean_f1_score = f1_scores.mean()

        data_num = self.total_count
        
        return classwise_acc, total_acc, f1_scores, mean_f1_score, data_num


class CCCLogger_torch:
    """Computes and stores the average and current value of CCC (Concordance Correlation Coefficient) for valence and arousal."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all stored values."""
        self.val_ccc_sum = torch.tensor(0.0, dtype=torch.float32, device='cpu')
        self.aro_ccc_sum = torch.tensor(0.0, dtype=torch.float32, device='cpu')
        self.count = torch.tensor(0.0, dtype=torch.float32, device='cpu')

    def update(self, val, pred_val, aro, pred_aro):
        """Update CCC values for valence and arousal with new batch data.
        
        Args:
            val (torch.Tensor): Ground truth valence, shape [batch, num_samples]
            pred_val (torch.Tensor): Predicted valence, shape [batch, num_samples]
            aro (torch.Tensor): Ground truth arousal, shape [batch, num_samples]
            pred_aro (torch.Tensor): Predicted arousal, shape [batch, num_samples]
        """
        val_ccc = self.CCC_loss_cal(pred_val, val)
        aro_ccc = self.CCC_loss_cal(pred_aro, aro)
        
        self.val_ccc_sum += val_ccc
        self.aro_ccc_sum += aro_ccc
        self.count += 1

    def gather(self):
        """Synchronize and aggregate values across multiple processes (for distributed training)."""
        dist.barrier()
        dist.all_reduce(self.val_ccc_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.aro_ccc_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.count, op=dist.ReduceOp.SUM)

    def final_score(self):
        """Compute final CCC scores after aggregation."""
        val_ccc = self.val_ccc_sum / (self.count + 1e-6)
        aro_ccc = self.aro_ccc_sum / (self.count + 1e-6)
        total_ccc = (val_ccc + aro_ccc) / 2  # Mean CCC

        return total_ccc, val_ccc, aro_ccc, self.count

    def CCC_loss_cal(self, x, y):
        """Compute Concordance Correlation Coefficient (CCC) between two tensors."""
        x_m = torch.mean(x, dim=-1, keepdim=True)  # Mean along sample dimension
        y_m = torch.mean(y, dim=-1, keepdim=True)
        x_s = torch.std(x, dim=-1, keepdim=True)
        y_s = torch.std(y, dim=-1, keepdim=True)

        vx = x - x_m
        vy = y - y_m

        rho = torch.mean(vx * vy, dim=-1) / (x_s * y_s + 1e-8)
        ccc = (2 * rho * x_s * y_s) / (torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2) + 1e-8)

        return ccc.mean()  # Mean across batch

class AU_Logger_torch:
    """Computes and stores the F1 scores for each AU and overall average F1 Score."""

    def __init__(self, num_au=12):
        self.num_au = num_au
        self.reset()

    def reset(self):
        """Reset all stored values."""
        self.true_positive = torch.zeros(self.num_au, dtype=torch.float32, device='cpu')
        self.false_positive = torch.zeros(self.num_au, dtype=torch.float32, device='cpu')
        self.false_negative = torch.zeros(self.num_au, dtype=torch.float32, device='cpu')

    def update(self, predictions, labels):
        """Update TP, FP, FN for each AU.
        
        Args:
            predictions (torch.Tensor): Model predictions (0 or 1), shape [batch, num_au]
            labels (torch.Tensor): Ground truth labels (0 or 1), shape [batch, num_au]
        """
        # Ensure predictions and labels are both binary (0 or 1) and integer type
        predictions = (predictions > 0.5).int()  # 확률값 → 이진값 변환
        labels = labels.int()  # 🔥 labels도 int로 변환하여 타입 맞추기

        # Update TP, FP, FN for each AU
        self.true_positive += (predictions & labels).sum(dim=0).float()
        self.false_positive += (predictions & ~labels).sum(dim=0).float()
        self.false_negative += (~predictions & labels).sum(dim=0).float()

    def gather(self):
        """Synchronize values across multiple GPUs (if using Distributed Training)."""
        dist.barrier()
        dist.all_reduce(self.true_positive, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.false_positive, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.false_negative, op=dist.ReduceOp.SUM)

    def final_score(self):
        """Compute F1 Scores for each AU and the overall mean F1 Score."""
        precision = self.true_positive / (self.true_positive + self.false_positive + 1e-12)
        recall = self.true_positive / (self.true_positive + self.false_negative + 1e-12)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)

        # NaN 방지 및 -0.0 → 0.0 변환
        f1_scores = torch.nan_to_num(f1_scores, nan=0.0, posinf=0.0, neginf=0.0).abs()

        # 평균 F1 Score 계산
        mean_f1_score = f1_scores.mean()

        return f1_scores, mean_f1_score

# JSD regularizer
def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    kl_div_p = F.kl_div(F.log_softmax(p, dim=1), F.softmax(m, dim=1), reduction='sum')
    kl_div_q = F.kl_div(F.log_softmax(q, dim=1), F.softmax(m, dim=1), reduction='sum')
    js_divergence = 0.5 * (kl_div_p + kl_div_q)
    return js_divergence


import torch.nn as nn
# NLA
class Integrated_Co_GA_Loss(nn.Module):
    def __init__(self, args, t_mean_x=0.50, t_mean_y=0.50, f_mean_x=0.30, f_mean_y=0.15, f_cm=0.80, f_std=0.85, t_cm=-0.50, t_std=0.75, t_lambda=1.0):
        super(Integrated_Co_GA_Loss, self).__init__()
        t_cm = args.eps * t_cm
        self.t_lambda = t_lambda
        self.t_mu = torch.tensor([t_mean_x, t_mean_y],dtype=torch.float).unsqueeze(1).to('cpu')
        self.f_mu = torch.tensor([f_mean_x, f_mean_y],dtype=torch.float).unsqueeze(1).to('cpu')
        self.t_cov = torch.tensor([[t_std, t_cm], [t_cm, t_std]],dtype=torch.float).to('cpu')
        self.f_cov = torch.tensor([[f_std, f_cm], [f_cm, f_std]],dtype=torch.float).to('cpu')
        self.w = torch.Tensor([1.0, 10.399988319803773, 16.23179290857716, 19.607905747632678, \
            1.8556467915720152, 2.225347712532647, 5.610554505356018, 1.0590043828089226]).to('cpu')
        
    def forward(self, inputs, targets):
        probs = torch.softmax(inputs, dim=1)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # wce_loss = F.cross_entropy(inputs, targets, self.w, reduction='none')

        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze()

        top2_probs = probs.topk(2, dim=1).values
        max_probs = top2_probs[:, 0]
        second_max_probs = top2_probs[:, 1]

        is_pt_max = max_probs == p_t
        
        p_negative = torch.where(is_pt_max, second_max_probs, max_probs)
        # 가우시안 분포 계산
        SV_flat = torch.stack([p_t, p_negative], axis=0)
        
        t_value = self.t_lambda*self.co_gau(SV_flat, self.t_cov, self.t_mu) 
        f_value = self.co_gau(SV_flat, self.f_cov, self.f_mu)

        value = torch.where(is_pt_max, t_value, f_value)
        adjusted_loss = ce_loss * value
        # adjusted_loss = ce_loss * value + wce_loss
        
        return adjusted_loss.mean()
    
    
    def co_gau(self, SV_flat, cov, mu):
        diff = SV_flat - mu
        inv_cov = torch.linalg.inv(cov)
        value = torch.exp(-0.5 * torch.matmul(torch.matmul(diff.t(), inv_cov), (diff)))
        value = torch.diagonal(value)
        return value
    


def exponential_scheduler(epoch, epochs, slope=-15, sch_bool=True):
    if sch_bool:
        return (1 - np.exp(slope * epoch / epochs))
    else:
        return 1
    
# CCC Loss
class CCC_loss(nn.Module):
    def __init__(self):
        super(CCC_loss, self).__init__()
        
    def forward(self, v_label, Vout, a_label, Aout):
        v_ccc = self.CCC_loss_cal(Vout,v_label) 
        a_ccc = self.CCC_loss_cal(Aout,a_label)
        ccc_loss = v_ccc + a_ccc
        mae_loss = (nn.MSELoss()(Vout, v_label) + nn.MSELoss()(Aout, a_label)) / 2
        return ccc_loss, mae_loss, v_ccc, a_ccc

    def CCC_loss_cal(self, x, y):
        x_m = torch.mean(x, dim=-1, keepdim=True)  
        y_m = torch.mean(y, dim=-1, keepdim=True)
        x_s = torch.std(x, dim=-1, keepdim=True)
        y_s = torch.std(y, dim=-1, keepdim=True)

        # std가 0이면 작은 값 추가
        x_s = torch.where(x_s == 0, torch.tensor(1e-6, device=x.device), x_s)
        y_s = torch.where(y_s == 0, torch.tensor(1e-6, device=y.device), y_s)

        vx = x - x_m
        vy = y - y_m

        rho = torch.mean(vx * vy, dim=-1) / (x_s * y_s + 1e-6)  # 분모가 0이 되지 않도록 보완
        ccc = (2 * rho * x_s * y_s) / ((torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2)) + 1e-6)

        return 1 - ccc  # CCC Loss 반환 (batch-wise)
    

def CCC_metric(x,y):
    x_m = torch.mean(x) # valence mean
    y_m = torch.mean(y) # arousal mean
    x_s = torch.std(x) # valence std
    y_s = torch.std(y) # arousal std
    vx = x - x_m # valence 편차
    vy = y - y_m # arousal 편차
    # valence 편차 * arousal 편차의 평균 / valence std * arousal std
    rho =  torch.mean(vx*vy) / (x_s*y_s)
    ccc = 2*rho*x_s*y_s/((torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))+1e-8)
    return ccc




def add_g(image_array, mean=0.0, var=30):
    std = var ** 0.5
    image_add = image_array + np.random.normal(mean, std, image_array.shape)
    image_add = np.clip(image_add, 0, 255).astype(np.uint8)
    return image_add

def flip_image(image_array):
    return cv2.flip(image_array, 1)