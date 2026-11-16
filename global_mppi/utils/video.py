"""Utilities for video recording using FFmpeg."""

import os
import subprocess
import tempfile
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class VideoRecorder:
    """Class for recording videos using FFmpeg."""

    def __init__(
        self,
        output_dir: str,
        width: int = 720,
        height: int = 480,
        fps: float = 30.0,
        name: str = "simulation",
    ):
        """Initialize the video recorder.

        Args:
            output_dir: Directory to save the video.
            width: Width of the video in pixels.
            height: Height of the video in pixels.
            fps: Frames per second.
            name: File name prefix; the start time is appended.
        """
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.fps = fps
        self.name = name

        self.ffmpeg_process = None
        self.video_path = None
        self.is_recording = False

    def start(self) -> bool:
        """Start recording the video.

        Returns:
            True if recording started successfully, False otherwise.
        """
        if self.is_recording:
            print("Warning: Recording already in progress")
            return True

        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Generate output path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = os.path.join(
            self.output_dir, f"{self.name}_{timestamp}.mp4"
        )

        # Check if FFmpeg is available
        try:
            # Test FFmpeg availability
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

            # Set up FFmpeg process
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file
                "-f",
                "rawvideo",  # Input format
                "-vcodec",
                "rawvideo",  # Input codec
                "-s",
                f"{self.width}x{self.height}",  # Frame dimensions
                "-pix_fmt",
                "rgb24",  # Pixel format
                "-r",
                str(self.fps),  # FPS
                "-i",
                "-",  # Input from pipe
                "-an",  # No audio
                "-vcodec",
                "h264",  # H.264 codec
                "-crf",
                "1",  # Quality (1=highest)
                "-preset",
                "slow",  # Encoding speed/compression tradeoff
                "-movflags",
                "+faststart",  # Optimize for web playback
                "-pix_fmt",
                "yuv420p",  # Standard compatible format
                "-profile:v",
                "high",  # H.264 profile
                "-tune",
                "film",  # Encoding optimization
                "-loglevel",
                "error",  # Suppress output except errors
                self.video_path,
            ]

            self.ffmpeg_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            self.is_recording = True
            print(f"Recording video to {self.video_path}")
            return True

        except (subprocess.SubprocessError, FileNotFoundError):
            print("Warning: FFmpeg not found. Video recording disabled.")
            self.is_recording = False
            return False

    def add_frame(self, frame: bytes) -> bool:
        """Add a frame to the video.

        Args:
            frame: Raw RGB frame data.

        Returns:
            True if the frame was added successfully, False otherwise.
        """
        if (
            not self.is_recording
            or self.ffmpeg_process is None
            or self.ffmpeg_process.stdin is None
        ):
            return False

        try:
            self.ffmpeg_process.stdin.write(frame)
            return True
        except (BrokenPipeError, IOError):
            print("Warning: Failed to write frame to video")
            self.is_recording = False
            return False

    def stop(self) -> bool:
        """Stop recording and finalize the video.

        Returns:
            True if the video was finalized successfully, False otherwise.
        """
        if not self.is_recording or self.ffmpeg_process is None:
            return False

        try:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()
            print(f"Video saved to {self.video_path}")
            self.is_recording = False
            return True
        except (subprocess.TimeoutExpired, BrokenPipeError, IOError) as e:
            print(f"Warning: Error finalizing video: {e}")
            # Try to terminate the process if it's still running
            try:
                self.ffmpeg_process.terminate()
            except:
                pass
            self.is_recording = False
            return False


def label_frame(frame: np.ndarray, text: str) -> np.ndarray:
    """Draw `text` in the top-left corner of an RGB frame."""
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(frame.shape[0] // 24, 12))
    draw.rectangle(draw.textbbox((8, 6), text, font=font), fill=(0, 0, 0))
    draw.text((8, 6), text, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def concat_videos(paths, output_path: str, delete_parts: bool = True) -> bool:
    """Join videos with the same size, fps and codec into one file (no re-encoding)."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as listing:
        listing.writelines(f"file '{os.path.abspath(p)}'\n" for p in paths)
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", listing.name, "-c", "copy", output_path], check=True)
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"Warning: could not combine videos ({e}); keeping the per-seed files")
        return False
    finally:
        os.remove(listing.name)
    if delete_parts:
        for p in paths:
            os.remove(p)
    print(f"Combined {len(paths)} videos into {output_path}")
    return True
