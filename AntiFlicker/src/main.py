import argparse
from pathlib import Path

from video_utils import VideoExtractor
from segmentation import SemanticSegmentation
from mask_analyzer import MaskAnalyzer
from smoother import MaskSmoother
from mask_comparator import MaskComparator
from loguru import logger

OUTPUT_DIRS = {
    "frames": "output/frames",
    "masks_raw": "output/masks_raw",
    "masks_smooth": "output/masks_smoothed",
    "plots": "output/plots",
    "videos": "output/videos",
}


def ensure_dirs():
    for path in OUTPUT_DIRS.values():
        Path(path).mkdir(parents=True, exist_ok=True)


def main():
    """Main entrypoint"""
    parser = argparse.ArgumentParser(description="Anti-flicker mask pipeline")

    parser.add_argument("--video", type=str, required=True, help="Path to input video")

    parser.add_argument(
        "--method",
        type=str,
        default="gaussian",
        choices=["gaussian", "median", "mean", "prob"],
        help="Smoothing method",
    )

    parser.add_argument("--window", type=int, default=8, help="Temporal window size")
    parser.add_argument("--sigma", type=float, default=None, help="Sigma for gaussian")

    parser.add_argument(
        "--output", type=str, default="result.mp4", help="Filename of output video"
    )

    args = parser.parse_args()
    ensure_dirs()

    extractor = VideoExtractor(output_dir=OUTPUT_DIRS["frames"])
    frames, fps = extractor.extract(args.video)

    logger.info("Running segmentation...")
    seg = SemanticSegmentation(output_dir=OUTPUT_DIRS["masks_raw"])
    seg.process(OUTPUT_DIRS["frames"])

    logger.info("Analyzing raw mask stability...")
    analyzer = MaskAnalyzer(OUTPUT_DIRS["masks_raw"], OUTPUT_DIRS["plots"])
    iou_raw = analyzer.analyze_stability()
    analyzer.plot_stability()

    logger.info("Smoothing masks...")
    smoother = MaskSmoother(OUTPUT_DIRS["masks_raw"], OUTPUT_DIRS["masks_smooth"])
    smoother.smooth(method=args.method, window=args.window, sigma=args.sigma)

    logger.info("Comparing...")
    cmp = MaskComparator(OUTPUT_DIRS["masks_raw"], OUTPUT_DIRS["masks_smooth"])
    iou_smooth = cmp.compare()
    cmp.plot_comparison(iou_raw, iou_smooth)

    improvement = (iou_smooth.mean() - iou_raw.mean()) / iou_raw.mean() * 100
    print(f"\n===== REPORT =====")
    print(f"Model: DeepLabv3")
    print(f"Method: {args.method} (window={args.window}, sigma={args.sigma})")
    print(f"Improvement: {improvement:+.2f}% IoU stability")
    print(f"Output saved to: output/videos/{args.output}")
    print("==================")


if __name__ == "__main__":
    main()
