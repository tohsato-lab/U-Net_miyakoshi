# パッケージのimport
import os.path as osp
from PIL import Image
import torch.utils.data as data

from utils.data_augumentation import *

def make_datapath_list(rootpath, phase):
    if phase == 'train':
        imgpath_template = osp.join(rootpath, 'train', '%s.tiff') #変更
        annopath_template = osp.join(rootpath, 'annotations/train_annotations', '%s.png') #変更

        # 訓練のファイルのID（ファイル名）を取得する
        train_id_names = osp.join(rootpath + 'segmentations/train.txt') #変更

        # 訓練データの画像ファイルとアノテーションファイルへのパスリストを作成
        train_img_list = list()
        train_anno_list = list()

        for line in open(train_id_names):
            file_id = line.strip()  # 空白スペースと改行を除去
            img_path = (imgpath_template % file_id)  # 画像のパス
            anno_path = (annopath_template % file_id)  # アノテーションのパス
            train_img_list.append(img_path)
            train_anno_list.append(anno_path)

        return train_img_list, train_anno_list
    
    elif phase == 'val':
        # 画像ファイルとアノテーションファイルへのパスのテンプレートを作成
        imgpath_template = osp.join(rootpath, 'val', '%s.tiff')
        annopath_template = osp.join(rootpath, 'annotations/val_annotations', '%s.png')

        # 訓練と検証、それぞれのファイルのID（ファイル名）を取得する
        val_id_names = osp.join(rootpath + 'segmentations/val.txt')

        # 検証データの画像ファイルとアノテーションファイルへのパスリストを作成
        val_img_list = list()
        val_anno_list = list()

        for line in open(val_id_names):
            file_id = line.strip()  # 空白スペースと改行を除去
            img_path = (imgpath_template % file_id)  # 画像のパス
            anno_path = (annopath_template % file_id)  # アノテーションのパス
            val_img_list.append(img_path)
            val_anno_list.append(anno_path)

        return val_img_list, val_anno_list
    
    elif phase == 'test':
        # 画像ファイルとアノテーションファイルへのパスのテンプレートを作成
        imgpath_template = osp.join(rootpath, 'test', '%s.tiff')
        annopath_template = osp.join(rootpath, 'annotations/test_annotations', '%s.png')

        # 訓練のファイルのID（ファイル名）を取得する
        test_id_names = osp.join(rootpath + 'segmentations/test.txt')

        # テストデータの画像ファイルとアノテーションファイルへのパスリストを作成
        test_img_list = list()
        test_anno_list = list()

        for line in open(test_id_names):
            file_id = line.strip()  # 空白スペースと改行を除去
            img_path = (imgpath_template % file_id)  # 画像のパス
            anno_path = (annopath_template % file_id)  # アノテーションのパス
            test_img_list.append(img_path)
            test_anno_list.append(anno_path)

        return test_img_list, test_anno_list
    

class DataTransform():
    """
    画像とアノテーションの前処理クラス。訓練時と検証時で異なる動作をする。
    画像のサイズをinput_size x input_sizeにする。
    訓練時はデータオーギュメンテーションする。

    Attributes
    ----------
    input_size : int
        リサイズ先の画像の大きさ。
    color_mean : (R, G, B)
        各色チャネルの平均値。
    color_std : (R, G, B)
        各色チャネルの標準偏差。
    """

    def __init__(self, input_size, color_mean, color_std): #スケーリングは削除する
        self.input_size = input_size
        self.color_mean = color_mean
        self.color_std = color_std
        self.data_transform = {
            'train': Compose([
                #Scale(scale=[0.8, 1.3]),  # 画像の拡大
                RandomRotation(angle=[-90, 90]),  # 回転
                GammaCorrection(gamma_range=(0.0, 2.0)),
                RandomMirror(),  # ランダムミラー
                Resize(input_size),  # リサイズ(input_size)
                Normalize_Tensor(color_mean, color_std)  # 色情報の標準化とテンソル化
            ]),
            'val': Compose([
                Resize(input_size),  # リサイズ(input_size)
                Normalize_Tensor(color_mean, color_std)  # 色情報の標準化とテンソル化
            ]),
            'test': Compose([
                Resize(input_size),  # リサイズ(input_size)
                Normalize_Tensor(color_mean, color_std)  # 色情報の標準化とテンソル化
            ])
        }

    def __call__(self, phase, img, anno_class_img):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img, anno_class_img)
        
class WDDD_WDDD2_Dataset(data.Dataset):
    """
    Attributes
    ----------
    img_list : リスト
        画像のパスを格納したリスト
    anno_list : リスト
        アノテーションへのパスを格納したリスト
    phase : 'train' or 'test'
        学習か訓練かを設定する。
    transform : object
        前処理クラスのインスタンス
    """

    def __init__(self, img_list, anno_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.img_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとアノテーションを取得
        '''
        img, anno_class_img = self.pull_item(index)
        return img, anno_class_img

    def pull_item(self, index):
        '''画像のTensor形式のデータ、アノテーションを取得する'''

        # 1. 画像読み込み
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)   # [高さ][幅][色RGB]

        # 2. アノテーション画像読み込み
        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path)   # [高さ][幅]

        # 3. 前処理を実施
        img, anno_class_img = self.transform(self.phase, img, anno_class_img)

        return img, anno_class_img
    

    