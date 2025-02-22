from pathlib import Path
from dataclasses import dataclass
import cv2
import json
import yaml
from rich.progress import track
from rich.console import Console
import logging
from typing import List, Tuple

console = Console()

@dataclass
class BBox:
    x: int
    y: int
    width: int
    height: int
    label: str
    
    @property
    def x2(self) -> int:
        return self.x + self.width
        
    @property
    def y2(self) -> int:
        return self.y + self.height
    
    def is_valid(self, img_shape: Tuple[int, int], min_size: int) -> bool:
        """检查边界框是否有效"""
        h, w = img_shape[:2]
        return (self.width >= min_size and 
                self.height >= min_size and
                0 <= self.x < w and 
                0 <= self.y < h and
                self.x2 <= w and 
                self.y2 <= h)

class ROICutter:
    def __init__(self, config_path: Path):
        self.config = yaml.safe_load(config_path.read_text())
        self.base_path = Path(__file__).parent
        self.setup_paths()
        
    def setup_paths(self):
        """设置路径并创建必要的目录"""
        self.frames_path = Path(self.config['paths']['frames_base'])
        self.rois_path = self.base_path / self.config['paths']['rois_output']
        self.rois_path.mkdir(exist_ok=True)
        
    def parse_bbox(self, label_data: dict, img_shape: Tuple[int, int]) -> BBox:
        """解析标注数据为边界框"""
        orig_w = label_data['original_width']
        orig_h = label_data['original_height']
        
        x = int(orig_w * label_data['x'] / 100)
        y = int(orig_h * label_data['y'] / 100)
        w = int(orig_w * label_data['width'] / 100)
        h = int(orig_h * label_data['height'] / 100)
        
        # 添加padding
        padding = self.config['roi'].get('padding', 0)
        if padding:
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(orig_w - x, w + 2 * padding)
            h = min(orig_h - y, h + 2 * padding)
            
        return BBox(x, y, w, h, label_data['rectanglelabels'][0])
    
    def process_image(self, image_data: dict) -> None:
        """处理单张图片的ROI提取"""
        # 构建图片路径
        img_rel_path = image_data['image'][21:]
        video_name = Path(img_rel_path).stem.split('_')[0]
        img_path = self.frames_path / video_name / Path(img_rel_path).name
        
        # 读取图片
        img = cv2.imread(str(img_path))
        if img is None:
            logging.warning(f"Failed to read image: {img_path}")
            return
            
        if not image_data.get('label'):
            return
            
        # 处理每个ROI区域
        for label in image_data['label']:
            bbox = self.parse_bbox(label, img.shape)
            
            # 验证边界框
            if not bbox.is_valid(img.shape, self.config['roi'].get('min_size', 10)):
                logging.warning(f"Invalid bbox in {img_path}: {bbox}")
                continue
                
            # 裁剪ROI
            roi = img[bbox.y:bbox.y2, bbox.x:bbox.x2]
            
            # 生成输出文件名
            output_name = f"{Path(img_rel_path).stem}-{bbox.label}-{bbox.x}_{bbox.y}.jpg"
            output_path = self.rois_path / output_name
            
            # 保存ROI
            cv2.imwrite(str(output_path), roi)
    
    def process_all(self):
        """处理所有图片"""
        # 加载标注文件
        json_path = self.base_path / 'label_detect.json'
        try:
            with open(json_path) as f:
                images = json.load(f)
        except Exception as e:
            console.print(f"[red]Error loading JSON file: {e}")
            return
            
        # 处理所有图片
        console.print("[green]Processing images...")
        for image_data in track(images, description="Cutting ROIs"):
            try:
                self.process_image(image_data)
            except Exception as e:
                logging.error(f"Error processing image {image_data.get('image')}: {e}")

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    cutter = ROICutter(Path(__file__).parent / 'config_roi.yaml')
    cutter.process_all()
    console.print("[green]ROI cutting completed!")

if __name__ == "__main__":
    main()
