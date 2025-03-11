from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict
import json
import random
import shutil
import zipfile
import yaml
from rich.progress import track
from rich.console import Console
from contextlib import contextmanager

console = Console()

@dataclass
class BoundingBox:
    cls: int
    x: float
    y: float
    w: float
    h: float
    
    def __str__(self):
        return f"{self.cls} {self.x} {self.y} {self.w} {self.h}"

@dataclass
class ImageAnnotation:
    image_path: Path
    boxes: List[BoundingBox]
    
class DatasetGenerator:
    def __init__(self, config_path: Path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        self.base_path = Path(__file__).parent
        self.labels_path = self.base_path / self.config['paths']['labels_detect']
        self.output_path = self.base_path / self.config['paths']['output_detect']
        self.frames_path = Path(self.config['paths']['frames_base'])
        
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
            shutil.rmtree(self.output_path, ignore_errors=True)
    
    def process_annotation(self, image_data: dict) -> ImageAnnotation:
        """处理单个图像标注数据"""
        image_path = Path(image_data['image'][21:])
        boxes = []
        
        if labels := image_data.get('label'):
            for label in labels:
                cls = self.config['detection']['labels'][label['rectanglelabels'][0]]
                x = label['x'] / 100 + label['width'] / 200
                y = label['y'] / 100 + label['height'] / 200
                w = label['width'] / 100
                h = label['height'] / 100
                boxes.append(BoundingBox(cls, x, y, w, h))
                
        return ImageAnnotation(image_path, boxes)
    
    def write_annotation(self, anno: ImageAnnotation):
        """写入标注文件"""
        label_file = self.labels_path / f"{anno.image_path.stem}.txt"
        with open(label_file, 'w') as f:
            for box in anno.boxes:
                f.write(f"{box}\n")
    
    def create_dataset(self, jsonpath):
        """生成数据集"""
        with self.create_temp_dirs():
            # 加载并处理标注
            with open(jsonpath) as f:
                labels_data = json.load(f)
            
            console.print("Processing annotations...", style="bold green")
            annotations = [self.process_annotation(img) for img in track(labels_data)]
            
            for anno in annotations:
                self.write_annotation(anno)
            
            # 收集数据集
            datasets = []
            for label_file in self.labels_path.glob('*.txt'):
                video_name = label_file.stem.split('_')[0]
                image_file = self.frames_path / video_name / f"{label_file.stem}.jpg"
                if image_file.exists():
                    datasets.append((image_file, label_file))
            
            # 分割数据集
            random.shuffle(datasets)
            split_idx = int(len(datasets) * self.config['dataset']['train_ratio'])
            train_set = datasets[:split_idx]
            val_set = datasets[split_idx:]
            
            # 复制文件
            console.print("Copying files...", style="bold green")
            for dataset, dir_name in [(train_set, 'train'), (val_set, 'val')]:
                output_dir = self.output_path / dir_name
                for image_file, label_file in track(dataset):
                    shutil.copy(image_file, output_dir)
                    shutil.copy(label_file, output_dir)
            
            # 创建zip文件
            console.print("Creating zip archive...", style="bold green")
            with zipfile.ZipFile(self.base_path / 'detect.zip', 'w') as zipf:
                for path in track(list(self.output_path.rglob('*'))):
                    if path.is_file():
                        zipf.write(path, path.relative_to(self.output_path))

def main():
    generator = DatasetGenerator(Path(__file__).parent / 'config.yaml')
    generator.create_dataset()
    console.print("Dataset generation completed!", style="bold green")

if __name__ == "__main__":
    main()
