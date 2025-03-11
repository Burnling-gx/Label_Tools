from pathlib import Path
import argparse
from rich.console import Console
import yaml
from video_to_dataset import VideoFrameExtractor
from generate_dataset_detect import DatasetGenerator
from generate_dataset_keypoint import KeypointDatasetGenerator
from cutroi import ROICutter
from view_dataset import DatasetViewer

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
        
    def generate_detection_dataset(self, json_path):
        """生成目标检测数据集"""
        console.print("\n[bold cyan]Generating detection dataset...[/bold cyan]")
        generator = DatasetGenerator(self.config)
        generator.create_dataset(json_path)
        
    def generate_keypoint_dataset(self, json_path, json_path2):
        """生成关键点数据集"""
        console.print("\n[bold cyan]Generating keypoint dataset...[/bold cyan]")
        generator = KeypointDatasetGenerator(self.config)
        generator.create_dataset(json_path, json_path2)
        
    def cut_rois(self, json_path):
        """裁剪ROI区域"""
        console.print("\n[bold cyan]Cutting ROIs from images...[/bold cyan]")
        cutter = ROICutter(self.config)
        cutter.process_all(json_path)
    
    def view_dataset(self, dataset_path, split):
        """查看数据集"""
        console.print("\n[bold cyan]Viewing dataset...[/bold cyan]")
        viewer = DatasetViewer(dataset_path, self.config)
        if split == 'all':
            viewer.browse_dataset(dataset_path, 'train')
            viewer.browse_dataset(dataset_path, 'val')
        else:
            viewer.browse_dataset(dataset_path, split)

def main():
    parser = argparse.ArgumentParser(description='Label Tools Manager')
    
    parser.add_argument('action', choices=['extract', 'detect', 'keypoint', 'roi', 'view'],
                      help='Action to perform')
    parser.add_argument('--video', help='Video path for frame extraction')
    parser.add_argument('--config', default='config_main.yaml',
                      help='Path to main config file')
    parser.add_argument('--json', help='Path to Label json file')
    parser.add_argument('--json2', help='Path to Label json file (keypoints of roi)')
    parser.add_argument('--dataset', help='Path to dataset directory')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'all'], help='Dataset split')
    
    args = parser.parse_args()
    
    try:
        manager = LabelToolManager(Path(__file__).parent / args.config)
        
        if args.action == 'extract':
            if not args.video:
                raise ValueError("Video path is required for frame extraction")
            manager.extract_frames(args.video)
            
        elif args.action == 'detect':
            if not args.json:
                raise ValueError("Label json file path is required for detection dataset generation")
            manager.generate_detection_dataset(args.json)
            
        elif args.action == 'keypoint':
            if not args.json:
                raise ValueError("Label json file path is required for keypoint dataset generation")
            if not args.json2:
                raise ValueError("Label json file path is required for keypoint dataset generation")
            manager.generate_keypoint_dataset(args.json, args.json2)
            
        elif args.action == 'roi':
            if not args.json:
                raise ValueError("Label json file path is required for ROI cutting")
            manager.cut_rois(args.json)

        elif args.action == 'view':
            if not args.dataset:
                raise ValueError("Dataset path is required for viewing")
            manager.view_dataset(args.dataset, args.split)
            
            
        console.print("\n[bold green]Operation completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"\n[bold red]Error: {str(e)}[/bold red]")
        raise

if __name__ == "__main__":
    main()
