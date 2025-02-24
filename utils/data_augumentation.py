# 第3章セマンティックセグメンテーションのデータオーギュメンテーション
# 注意　アノテーション画像はカラーパレット形式（インデックスカラー画像）となっている。

# パッケージのimport
import torch
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np


class Compose(object):
    """引数transformに格納された変形を順番に実行するクラス
       対象画像とアノテーション画像を同時に変換させます。 
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, anno_class_img):
        for t in self.transforms:
            img, anno_class_img = t(img, anno_class_img)
        return img, anno_class_img


class Scale(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img, anno_class_img):

        width = img.size[0]  # img.size=[幅][高さ]
        height = img.size[1]  # img.size=[幅][高さ]

        # 拡大倍率をランダムに設定
        scale = np.random.uniform(self.scale[0], self.scale[1])

        scaled_w = int(width * scale)  # img.size=[幅][高さ]
        scaled_h = int(height * scale)  # img.size=[幅][高さ]

        # 画像のリサイズ
        img = img.resize((scaled_w, scaled_h), Image.BICUBIC)

        # アノテーションのリサイズ
        anno_class_img = anno_class_img.resize(
            (scaled_w, scaled_h), Image.NEAREST)

        # 画像を元の大きさに
        # 切り出し位置を求める
        if scale > 1.0:
            left = scaled_w - width
            left = int(np.random.uniform(0, left))

            top = scaled_h-height
            top = int(np.random.uniform(0, top))

            img = img.crop((left, top, left+width, top+height))
            anno_class_img = anno_class_img.crop(
                (left, top, left+width, top+height))

        else:
            # input_sizeよりも短い辺はpaddingする
            p_palette = anno_class_img.copy().getpalette()

            img_original = img.copy()
            anno_class_img_original = anno_class_img.copy()

            pad_width = width-scaled_w
            pad_width_left = int(np.random.uniform(0, pad_width))

            pad_height = height-scaled_h
            pad_height_top = int(np.random.uniform(0, pad_height))


            img = Image.new(img.mode, (width, height), (0))
            img.paste(img_original, (pad_width_left, pad_height_top))
            
            anno_class_img = Image.new(anno_class_img.mode, (width, height), (0,0,0))
            anno_class_img.paste(anno_class_img_original,
                                 (pad_width_left, pad_height_top))
            anno_class_img.putpalette(p_palette)

        return img, anno_class_img


class RandomRotation(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img, anno_class_img):

        # 回転角度を決める
        rotate_angle = (np.random.uniform(self.angle[0], self.angle[1]))

        # 回転
        img = img.rotate(rotate_angle, Image.BILINEAR)
        anno_class_img = anno_class_img.rotate(rotate_angle, Image.NEAREST)

        return img, anno_class_img


class RandomMirror(object):
    """50%の確率で左右反転させるクラス"""

    def __call__(self, img, anno_class_img):
        if np.random.randint(2):
            img = ImageOps.mirror(img)
            anno_class_img = ImageOps.mirror(anno_class_img)
        return img, anno_class_img


class Resize(object):
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, img, anno_class_img):
        return self.apply_resize(img, anno_class_img, self.input_size)

    @staticmethod
    def apply_resize(img, anno_class_img, input_size):
        # 元画像のアスペクト比を維持
        w, h = img.size
        if w > h:
            new_h = input_size
            new_w = int(w * (input_size / h))
        else:
            new_w = input_size
            new_h = int(h * (input_size / w))

        # リサイズ
        img = img.resize((new_w, new_h), Image.BICUBIC)
        anno_class_img = anno_class_img.resize((new_w, new_h), Image.NEAREST)

        return img, anno_class_img


class Normalize_Tensor(object):
    def __init__(self, color_mean, color_std):
        self.color_mean = color_mean
        self.color_std = color_std

    def __call__(self, img, anno_class_img):

        # PIL画像をTensorに。大きさは最大1に規格化される
        img = transforms.functional.to_tensor(img)

        # 色情報の標準化
        img = transforms.functional.normalize(
            img, self.color_mean, self.color_std)

        # アノテーション画像をNumpyに変換
        anno_class_img = np.array(anno_class_img)  # [高さ][幅]

        # 'ambigious'には255が格納されているので、0の背景にしておく
        index = np.where(anno_class_img == 255)
        anno_class_img[index] = 0

        # アノテーション画像をTensorに
        anno_class_img = torch.from_numpy(anno_class_img)

        return img, anno_class_img

class Brightness(object):
    """ランダムに輝度を変更するクラス"""

    def __init__(self, brightness_range=(0.8, 1.5)):
        """Arg:
            brightness_range(tuple): 輝度変更の範囲
        """
        self.brightness_range = brightness_range

    def __call__(self, img, anno_class_img):
        """
        画像とアノテーション画像を受け取り、画像の輝度をランダムに変更して返す

        Args:
            img (PIL.Image): 入力画像
            anno_class_img (PIL.Image): アノテーション画像

        Returns:
            PIL.Image: 輝度変更後の画像
            PIL.Image: 変更なしのアノテーション画像 (輝度変更は影響しない)
        """
        brightness_factor = np.random.uniform(self.brightness_range[0],
                                              self.brightness_range[1])
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)

        return img, anno_class_img
    
from PIL import Image
import numpy as np

class GammaCorrection(object):
    """ランダムにガンマ補正を行うクラス"""

    def __init__(self, gamma_range=(0.0, 2.0)):
        """
        Args:
            gamma_range(tuple): ガンマ補正の範囲
        """
        self.gamma_range = gamma_range

    def __call__(self, img, anno_class_img):
        """
        画像とアノテーション画像を受け取り、画像のガンマ補正をランダムに変更して返す

        Args:
            img (PIL.Image): 入力画像
            anno_class_img (PIL.Image): アノテーション画像

        Returns:
            PIL.Image: ガンマ補正後の画像
            PIL.Image: 変更なしのアノテーション画像 (ガンマ補正は影響しない)
        """
        gamma_factor = np.random.uniform(self.gamma_range[0], self.gamma_range[1])

        # 画像をNumPy配列に変換
        img_np = np.array(img)
        max_val = np.max(img_np)  # 画像の最大値を取得

        # 最大値で正規化してからガンマ補正を適用
        img_np = max_val * (img_np / max_val) ** 1/gamma_factor 
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)  # [0, 255]の範囲に戻す

        # NumPy配列をPIL画像に変換
        img = Image.fromarray(img_np)

        return img, anno_class_img
    
class RandomGammaCorrection(object):
    def __init__(self, min_gamma=0, max_gamma=2):
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma

    @staticmethod
    def gamma_correction(img: np.ndarray, gamma):
        max_val = np.max(img)
        img = max_val * (img / max_val) ** (1 / gamma)
        return img.astype(np.uint8)

    def __call__(self, kwargs: dict) -> dict:
        gamma = np.random.uniform(self.min_gamma, self.max_gamma)
        return {
            key: np.asarray(kwargs[key]) if 'label' in key else self.gamma_correction(np.asarray(kwargs[key]), gamma)
            for key in kwargs.keys()
        }