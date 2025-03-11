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
    image_shape: List[int]

@dataclass
class KeypointAnnotation:
    image_path: Path
    keypoints: List[Keypoint]
    bbox: BoundingBox

    def __str__(self):
        return f"{self.bbox} {' '.join(map(str, self.keypoints))}"

class KeypointDatasetGenerator:
    def __init__(self, config_path: Path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        self.base_path = Path(__file__).parent
        self.rois_path = self.base_path / self.config['paths']['rois_output']
        self.labels_path = self.base_path / self.config['paths']['labels_keypoint']
        self.output_path = self.base_path / self.config['paths']['output_keypoint']
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
            
    def process_annotation_keypoints(self, image_data: dict) -> KeypointAnnotation:
        """处理单个图像的关键点标注"""
        image_path = Path(image_data['img'][21:])
        frame_name = image_path.stem.split('-')[0]

        keypoints = []

        if kps := image_data.get('kp-1'):
            if len(kps) != 4:
                raise ValueError("Invalid number of keypoints")
            # kps = [(label, x_pix, y_pix]
            kps = [(kp['keypointlabels'][0], kp['x']*kp['original_width']/100, kp['y']*kp['original_height']/100) for kp in kps]
            # print(kps)
            # Check that all required keypoints are present exactly once
            keypoint_counts = {"kp1": 0, "kp2": 0, "kp3": 0, "kp4": 0}
            for kp in kps:
                label = kp[0]
                if label in keypoint_counts:
                    keypoint_counts[label] += 1
                else:
                    raise ValueError(f"Unknown keypoint label: {label}, image: {frame_name}")

            for label, count in keypoint_counts.items():
                if count != 1:
                    raise ValueError(f"Expected 1 {label}, found {count}, image: {frame_name}")

            # Convert to dict for easier access
            keypoints_dict = {kp[0]: {"x": kp[1], "y": -kp[2]} for kp in kps}

            # Compute center point
            center_x = sum(kp[1] for kp in kps) / len(kps)
            center_y = sum(kp[2] for kp in kps) / len(kps)

            # Calculate angles for each keypoint
            for kp in kps:
                label, x, y = kp[0], kp[1], kp[2]
                dx = x - center_x
                dy = y - center_y
                angle = math.atan2(dx, dy)
                # Convert to [0, 2π]
                angle = angle if angle >= 0 else angle + 2 * math.pi
                keypoints_dict[label]["angle"] = angle
            
            # print(keypoints_dict)
            # Check if points are in counterclockwise order: kp1, kp2, kp3, kp4
            expected_order = ["kp1", "kp2", "kp3", "kp4"]
            sorted_kps = sorted(expected_order, key=lambda k: keypoints_dict[k]["angle"])
            if ''.join(expected_order) not in ''.join(sorted_kps)*2:
                raise ValueError(f"Keypoints not in counterclockwise order. Found: {sorted_kps}, image: {frame_name}")

            for kp in kps:
                keypoints.append(Keypoint(kp[1], kp[2]))
        
        anno = KeypointAnnotation(
            image_path=image_path,
            keypoints=keypoints,
            bbox=None
        )
        
        return anno
    
    def write_annotation(self, anno: List[KeypointAnnotation]):
        """写入标注文件"""
        # print(anno)
        if not anno:
            return
        frame_name = anno[0].image_path.stem.split('-')[0]
        label_file = self.labels_path / f"{frame_name}.txt"
        with open(label_file, 'w') as f:
            for single_anno in anno:
                f.write(str(single_anno))
                f.write('\n')
                # print(single_anno)

    def process_annotation_detect(self, image_data: dict) -> ImageAnnotation:
        """处理单个图像标注数据"""
        image_path = Path(image_data['image'][21:])
        image_shape = [1080, 1440]
        boxes = []

        if labels := image_data.get('label'):
            for label in labels:
                cls = self.config['detection']['labels'][label['rectanglelabels'][0]]
                x = label['x'] / 100 + label['width'] / 200
                y = label['y'] / 100 + label['height'] / 200
                w = label['width'] / 100
                h = label['height'] / 100
                boxes.append(BoundingBox(cls, x, y, w, h))
                
        return ImageAnnotation(image_path, boxes, image_shape)
    
    def create_dataset(self, json_path, json_path2):
        """生成数据集"""
        with self.create_temp_dirs():
            # 加载并处理标注
            with open(json_path) as f:
                labels_data = json.load(f)
            with open(json_path2) as f:
                labels_data_kp = json.load(f)
            
            console.print("Processing annotations...", style="bold green")

            self.annotations_detect = {Path(img['image'][21:]).stem.split('-')[0] : self.process_annotation_detect(img) for img in track(labels_data)}
            annotations = {}
            
            for img in track(labels_data_kp):
                frame_name = Path(img['img'][21:]).stem.split('-')[0]
                frame_id = int(Path(img['img'][21:]).stem.split('-')[-1]) - 1
                bbox = self.annotations_detect[frame_name].boxes[frame_id] # xywhn
                anno = self.process_annotation_keypoints(img) # xy
                anno_kp = [Keypoint((kp.x+(bbox.x-bbox.w/2)*1440)/1440, (kp.y+(bbox.y-bbox.h/2)*1080)/1080) for kp in anno.keypoints]
                anno.keypoints = anno_kp
                anno.bbox = self.annotations_detect[frame_name].boxes[frame_id]
                if frame_name not in annotations:
                    annotations[frame_name] = list()

                annotations[frame_name].append(anno)
                
            
            for anno in annotations:
                self.write_annotation(annotations[anno])
            
            # 收集数据集
            datasets = []
            exclude_videos = set(self.config['dataset']['exclude_videos'])
            
            for label_file in self.labels_path.glob('*.txt'):
                if label_file.stem.split('_')[0] in exclude_videos:
                    continue
                video_name = label_file.stem.split('_')[0]
                image_file = self.frames_path / video_name / f"{label_file.stem}.jpg"
                # print(label_file, image_file)
                if image_file.exists():
                    datasets.append((image_file, label_file))

            # print(datasets)
            
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
                    # self.resize_image(image_file, resized_image_path)
                    shutil.copy(label_file, output_dir)
                    shutil.copy(image_file, output_dir)
            
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