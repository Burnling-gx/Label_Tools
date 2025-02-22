from pathlib import Path
import cv2
import yaml
import argparse
from rich.progress import Progress
from rich.console import Console
from rich import print as rprint
import logging
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np

@dataclass
class VideoControl:
    """视频控制状态"""
    wait: int
    current_frame: int
    total_frames: int
    is_playing: bool = True

class VideoFrameExtractor:
    def __init__(self, config_path: Path):
        """初始化视频帧提取器"""
        self.config = yaml.safe_load(config_path.read_text())
        self.console = Console()
        self.video_name = None
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def create_output_dir(self, video_path: Path) -> Path:
        """创建输出目录"""
        self.video_name = video_path.stem.split('_')[0]
        output_dir = Path(self.config['paths']['frames_base']) / self.video_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
        
    def display_controls(self):
        """显示控制说明"""
        controls = self.config['video']['controls']
        rprint("\n[bold cyan]Control Keys:[/bold cyan]")
        for action, key in controls.items():
            rprint(f"[green]{key}[/green]: {action.replace('_', ' ').title()}")
    
    def process_video(self, video_path: str):
        """处理视频文件"""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        output_dir = self.create_output_dir(video_path)
        cap = cv2.VideoCapture(str(video_path))
        
        control = VideoControl(
            wait=self.config['video']['default_wait'],
            current_frame=0,
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        )
        
        window_name = self.config['video']['window_name']
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        self.display_controls()
        
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, control.current_frame)
            ret, frame = cap.read()
            
            if not ret:
                self.console.print("End of video reached", style="bold red")
                break
                
            # 显示帧计数
            frame_info = f"Frame: {control.current_frame}/{control.total_frames}"
            frame_infoed = frame.copy()
            cv2.putText(frame_infoed, frame_info, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(window_name, frame_infoed)
            key = cv2.waitKey(control.wait) & 0xFF
            
            if not self.handle_key_press(key, frame, output_dir, control):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
    def handle_key_press(self, key: int, frame: np.ndarray, 
                        output_dir: Path, control: VideoControl) -> bool:
        """处理按键事件"""
        controls = self.config['video']['controls']
        
        if chr(key) == controls['quit']:
            return False
            
        elif chr(key) == controls['play_pause']:
            control.is_playing = not control.is_playing
            control.wait = self.config['video']['default_wait'] if control.is_playing else 0
            
        elif chr(key) == controls['save_frame']:
            output_path = output_dir / f"{self.video_name}_f_{control.current_frame}.jpg"
            cv2.imwrite(str(output_path), frame)
            self.console.print(f"Saved frame to {output_path}", style="green")
            
        elif chr(key) == controls['prev_frame']:
            control.current_frame = max(0, control.current_frame - 1)
            
        elif chr(key) == controls['next_frame']:
            control.current_frame = min(control.total_frames - 1, control.current_frame + 1)
            
        elif chr(key) == controls['fast_forward']:
            control.current_frame = min(control.total_frames - 1, control.current_frame + 10)
            
        elif chr(key) == controls['rewind']:
            control.current_frame = max(0, control.current_frame - 10)
            
        elif control.is_playing and control.current_frame < control.total_frames - 1:
            control.current_frame += 1
            
        return True

def main():
    parser = argparse.ArgumentParser(description='Video Frame Extractor')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--config', default='config_video.yaml',
                      help='Path to config file')
    args = parser.parse_args()
    
    try:
        extractor = VideoFrameExtractor(Path(__file__).parent / args.config)
        extractor.process_video(args.video_path)
    except Exception as e:
        logging.error(f"Error processing video: {e}")
        raise

if __name__ == "__main__":
    main()