from pathlib import Path
import argparse
from rich.console import Console
import yaml
from video_to_dataset import VideoFrameExtractor
from generate_dataset_detect import DatasetGenerator
from generate_dataset_keypoint import KeypointDatasetGenerator
from cutroi import ROICutter

console = Console()

class LabelToolManager:
    def __init__(self, config_path: Path):
        """初始化标注工具管理器"""
        self.config = config_path
        self.base_path = Path(__file__).parent
        
    def extract_frames(self, video_path: str):
        """从视频中提取帧"""
        console.print("\n[bold cyan]Starting video frame extraction...[/bold cyan]")
        extractor = VideoFrameExtractor(self.config)
        extractor.process_video(video_path)
        
    def generate_detection_dataset(self):
        """生成目标检测数据集"""
        console.print("\n[bold cyan]Generating detection dataset...[/bold cyan]")
        generator = DatasetGenerator(self.config)
        generator.create_dataset()
        
    def generate_keypoint_dataset(self):
        """生成关键点数据集"""
        console.print("\n[bold cyan]Generating keypoint dataset...[/bold cyan]")
        generator = KeypointDatasetGenerator(self.config)
        generator.create_dataset()
        
    def cut_rois(self):
        """裁剪ROI区域"""
        console.print("\n[bold cyan]Cutting ROIs from images...[/bold cyan]")
        cutter = ROICutter(self.config)
        cutter.process_all()

def main():
    parser = argparse.ArgumentParser(description='Label Tools Manager')
    
    parser.add_argument('action', choices=['extract', 'detect', 'keypoint', 'roi'],
                      help='Action to perform')
    parser.add_argument('--video', help='Video path for frame extraction')
    parser.add_argument('--config', default='config_main.yaml',
                      help='Path to main config file')
    
    args = parser.parse_args()
    
    try:
        manager = LabelToolManager(Path(__file__).parent / args.config)
        
        if args.action == 'extract':
            if not args.video:
                raise ValueError("Video path is required for frame extraction")
            manager.extract_frames(args.video)
            
        elif args.action == 'detect':
            manager.generate_detection_dataset()
            
        elif args.action == 'keypoint':
            manager.generate_keypoint_dataset()
            
        elif args.action == 'roi':
            manager.cut_rois()
            
        console.print("\n[bold green]Operation completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"\n[bold red]Error: {str(e)}[/bold red]")
        raise

if __name__ == "__main__":
    main()
