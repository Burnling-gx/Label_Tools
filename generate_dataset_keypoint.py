from pathlib import Path
from dataclasses import dataclass
from typing import List
import json
import random
import shutil
import yaml
from rich.progress import track
from rich.console import Console
from contextlib import contextmanager
from zipfile import ZipFile
import cv2
import math

console = Console()

@dataclass
class Keypoint:
    x: float
    y: float
    
    def __str__(self):
        return f"{self.x} {self.y}"
        
    def angle_from_center(self, center_x: float, center_y: float) -> float:
        """计算关键点相对于中心点的角度"""
        dx = self.x - center_x
        dy = self.y - center_y
        angle = math.atan2(dy, dx)
        # 将角度转换到 [0, 2π] 范围
        return angle if angle >= 0 else angle + 2 * math.pi

@dataclass
class KeypointAnnotation:
    image_path: Path
    keypoints: List[Keypoint]
    bbox: List[float]
    cls: int = 0
    
    def sort_keypoints_by_angle(self) -> None:
        """按照向量角度排序关键点"""
        if not self.keypoints:
            return
            
        # 计算中心点
        center_x = sum(kp.x for kp in self.keypoints) / len(self.keypoints)
        center_y = sum(kp.y for kp in self.keypoints) / len(self.keypoints)
        
        # 根据角度排序
        self.keypoints.sort(
            key=lambda kp: kp.angle_from_center(center_x, center_y)
        )
    
    def __str__(self):
        # 先排序再转字符串
        self.sort_keypoints_by_angle()
        kps_str = " ".join(str(kp) for kp in self.keypoints)
        bbox_str = " ".join(map(str, self.bbox))
        return f"{self.cls} {bbox_str} {kps_str}"

class KeypointDatasetGenerator:
    def __init__(self, config_path: Path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        self.base_path = Path(__file__).parent
        self.rois_path = self.base_path / self.config['paths']['rois_output']
        self.labels_path = self.base_path / self.config['paths']['labels_keypoint']
        self.output_path = self.base_path / self.config['paths']['output_keypoint']
        
    @contextmanager
    def create_temp_dirs(self):
        """创建临时目录并在使用完后清理"""
        try:
            self.labels_path.mkdir(exist_ok=True)
            self.output_path.mkdir(exist_ok=True)
            (self.output_path / 'train').mkdir(exist_ok=True)
            (self.output_path / 'val').mkdir(exist_ok=True)
            yield
        finally:
            shutil.rmtree(self.labels_path, ignore_errors=True)
            
    def process_annotation(self, image_data: dict) -> KeypointAnnotation:
        """处理单个图像的关键点标注"""
        image_path = Path(image_data['img'][21:])
        keypoints = []
        
        if kps := image_data.get('kp-1'):
            for kp in kps:
                x = kp['x'] / 100
                y = kp['y'] / 100
                keypoints.append(Keypoint(x, y))
                        
        anno = KeypointAnnotation(
            image_path=image_path,
            keypoints=keypoints,
            bbox=self.config['keypoint']['default_bbox'],
            cls=0
        )
        
        return anno
    
    def write_annotation(self, anno: KeypointAnnotation):
        """写入标注文件"""
        if not anno.keypoints:
            return
            
        label_file = self.labels_path / f"{anno.image_path.stem}.txt"
        with open(label_file, 'w') as f:
            f.write(f"{anno}\n")
    
    def resize_image(self, image_path: Path, output_path: Path):
        """将图片缩放至 128x128 并保存"""
        image = cv2.imread(str(image_path))
        resized_image = cv2.resize(image, (128, 128))
        cv2.imwrite(str(output_path), resized_image)
    
    def create_dataset(self):
        """生成数据集"""
        with self.create_temp_dirs():
            # 加载并处理标注
            with open(self.base_path / 'label_keypoint.json') as f:
                labels_data = json.load(f)
            
            console.print("Processing annotations...", style="bold green")
            annotations = [self.process_annotation(img) for img in track(labels_data)]
            
            for anno in annotations:
                self.write_annotation(anno)
            
            # 收集数据集
            datasets = []
            exclude_videos = set(self.config['dataset']['exclude_videos'])
            
            for label_file in self.labels_path.glob('*.txt'):
                if label_file.stem.split('_')[0] in exclude_videos:
                    continue
                    
                image_file = self.rois_path / f"{label_file.stem}.jpg"
                if image_file.exists():
                    datasets.append((image_file, label_file))
            
            # 分割数据集
            random.shuffle(datasets)
            split_idx = int(len(datasets) * self.config['dataset']['train_ratio'])
            train_set = datasets[:split_idx]
            val_set = datasets[split_idx:]
            
            # 复制文件并缩放图片
            console.print("Copying and resizing files...", style="bold green")
            for dataset, dir_name in [(train_set, 'train'), (val_set, 'val')]:
                output_dir = self.output_path / dir_name
                for image_file, label_file in track(dataset):
                    resized_image_path = output_dir / image_file.name
                    self.resize_image(image_file, resized_image_path)
                    shutil.copy(label_file, output_dir)
            
            # 打包数据集
            console.print("Creating zip archive...", style="bold green")
            with ZipFile(self.base_path / 'keypoints.zip', 'w') as zipf:
                for folder_name in ['train', 'val']:
                    folder_path = self.output_path / folder_name
                    for file_path in folder_path.rglob('*'):
                        zipf.write(file_path, file_path.relative_to(self.output_path))

def main():
    generator = KeypointDatasetGenerator(Path(__file__).parent / 'config_keypoint.yaml')
    generator.create_dataset()
    console.print("Dataset generation completed!", style="bold green")

if __name__ == "__main__":
    main()