import argparse
import os
import cv2
import numpy as np
from pathlib import Path
import yaml
import tempfile
import zipfile
import shutil

class DatasetViewer:
    def __init__(self, dataset_path, config_path=None):
        self.dataset_path = Path(dataset_path)
        
        # 加载配置文件（如果提供）
        self.class_names = ["object"]  # 默认类名
        self.keypoint_names = ["kp1", "kp2", "kp3", "kp4"]  # 默认关键点名称
        
        if config_path:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                if "detection" in config and "labels" in config["detection"]:
                    self.class_names = list(config["detection"]["labels"].keys())
                if "keypoints" in config and "labels" in config["keypoints"]:
                    self.keypoint_names = list(config["keypoints"]["labels"].keys())
    
    def _draw_label(self, img, label, color=(0, 255, 0), font_scale=0.5):
        """在图像上绘制标签文本"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, font_scale, 1)[0]
        cv2.putText(img, label, (0, text_size[1]), font, font_scale, color, 1)
    
    def draw_bbox(self, img, bbox, class_idx):
        """绘制边界框及类别标签"""
        h, w, _ = img.shape
        x_center, y_center, width, height = bbox
        
        # 转换为像素坐标
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        
        # 绘制边界框
        color = (0, 255, 0)  # 绿色
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # 绘制类别标签
        class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"class_{class_idx}"
        label = f"{class_name}"
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img
    
    def draw_keypoints(self, img, keypoints):
        """绘制关键点及连接线"""
        h, w, _ = img.shape
        points = []
        
        # 绘制关键点
        for i in range(0, len(keypoints), 2):
            if i+1 < len(keypoints):
                x, y = keypoints[i], keypoints[i+1]
                x_px, y_px = int(x * w), int(y * h)
                points.append((x_px, y_px))
                
                # 绘制关键点
                color = (0, 0, 255)  # 红色
                cv2.circle(img, (x_px, y_px), 5, color, -1)
                
                # 添加关键点标签
                kp_idx = i // 2
                if kp_idx < len(self.keypoint_names):
                    kp_name = self.keypoint_names[kp_idx]
                    cv2.putText(img, kp_name, (x_px + 5, y_px + 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 连接关键点（如果有4个点，则按顺序连接形成多边形）
        # if len(points) == 4:
        #     cv2.line(img, points[0], points[1], (255, 0, 0), 2)
        #     cv2.line(img, points[1], points[2], (255, 0, 0), 2)
        #     cv2.line(img, points[2], points[3], (255, 0, 0), 2)
        #     cv2.line(img, points[3], points[0], (255, 0, 0), 2)
        
        return img
    
    def visualize_sample(self, img_path, label_path=None):
        """可视化单个样本及其标注"""
        if not os.path.exists(img_path):
            print(f"图像不存在: {img_path}")
            return None

        print(img_path, label_path)    
        img = cv2.imread(img_path)
    
        if img is None:
            print(f"无法读取图像: {img_path}")
            return None
        
        # 添加图像路径标签
        self._draw_label(img, f"Image: {Path(img_path).name}")
        
        if label_path and os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if not parts:
                    continue
                    
                class_idx = int(parts[0])
                bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                
                # 绘制边界框
                img = self.draw_bbox(img, bbox, class_idx)
                
                # 如果有关键点数据
                if len(parts) > 5:
                    keypoints = [float(p) for p in parts[5:]]
                    img = self.draw_keypoints(img, keypoints)
                
        return img
    
    def browse_dataset(self, dataset_path, split='train'):
        """浏览数据集中的样本，支持直接查看zip文件"""
        
        # Check if the dataset is a zip file
        if str(dataset_path).endswith('.zip'):
            print(f"正在处理zip文件: {dataset_path}")
            
            # Create temporary directory for extraction
            temp_dir = tempfile.mkdtemp()
            try:
                # Extract only the necessary files
                with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
                    # Get all files in the zip
                    all_files = zip_ref.namelist()
                    
                    # Filter to only extract the relevant split directory
                    split_files = [f for f in all_files if f.startswith(f"{split}/")]
                    if not split_files:
                        print(f"在zip文件中没有找到 {split} 分割目录")
                        return
                    
                    print(f"从zip中提取 {split} 目录...")
                    for file in split_files:
                        zip_ref.extract(file, temp_dir)
                
                # Set the dataset path to the temporary directory + split
                split_dir = Path(temp_dir) / split
            except Exception as e:
                print(f"解压文件时出错: {e}")
                shutil.rmtree(temp_dir)
                return
        else:
            # Regular directory handling
            split_dir = Path(dataset_path) / split
            if not split_dir.exists():
                print(f"数据集分割目录不存在: {split_dir}")
                return
        
        # 获取所有图像文件
        img_extensions = ['.jpg', '.jpeg', '.png']
        img_files = []
        for ext in img_extensions:
            img_files.extend(list(split_dir.glob(f"*{ext}")))
        img_files.sort()
        
        if not img_files:
            print(f"在 {split_dir} 中没有找到图像文件")
            if str(dataset_path).endswith('.zip'):
                shutil.rmtree(temp_dir)  # Clean up temp directory
            return
        
        print(f"找到 {len(img_files)} 个图像文件")
        
        current_idx = 0
        while current_idx < len(img_files):
            img_path = img_files[current_idx]
            label_path = img_path.with_suffix('.txt')
            
            img = self.visualize_sample(str(img_path), str(label_path) if label_path.exists() else None)
            if img is None:
                current_idx += 1
                continue
            
            # 显示图像，包含导航指令
            cv2.putText(img, "Press q to quit, s to save", 
                         (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Create a single window name
            window_name = "Dataset Viewer"
            cv2.imshow(window_name, img)
            # Update window title to show current image info
            cv2.setWindowTitle(window_name, f"{img_path.name} ({current_idx+1}/{len(img_files)})")
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):  # 退出
                break
            elif key == 83:  # 右箭头 - 下一张
                current_idx = min(current_idx + 1, len(img_files) - 1)
            elif key == 81:  # 左箭头 - 上一张
                current_idx = max(current_idx - 1, 0)
            elif key == ord('s'):  # 保存当前图像
                save_path = f"visualization_{Path(img_path).stem}.jpg"
                cv2.imwrite(save_path, img)
                print(f"已保存图像到: {save_path}")
        
        # Clean up temporary directory if using zip
        if str(dataset_path).endswith('.zip'):
            print(f"清理临时文件...")
            shutil.rmtree(temp_dir)
        

def main():
    parser = argparse.ArgumentParser(description='YOLO 数据集可视化工具')
    parser.add_argument('dataset_path', type=str, help='数据集根目录路径')
    parser.add_argument('--config', type=str, help='配置文件路径（可选）')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'], help='数据集分割（默认: train）')
    args = parser.parse_args()
    
    viewer = DatasetViewer(args.dataset_path, args.config)
    viewer.browse_dataset(args.split)

if __name__ == "__main__":
    main()
