"""Command line interface for the Fire Tracking pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .calibrate_logs import calibrate_from_log, summarize_log_and_run
from .calibrate_mocap import export_mocap_calibration
from .clicks import click_status, serve_click_ui
from .detect import run_detection
from .format_527 import normalize_dataset
from .triangulate_527 import run_triangulation
from .webui import serve_webui


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _add_only(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="Camera label to process. May be repeated, e.g. --only ardu_run1_pixel9.",
    )


def _cmd_format(args: argparse.Namespace) -> int:
    normalize_dataset(
        args.data_root,
        args.out_root,
        only=args.only,
        overwrite=args.overwrite,
    )
    return 0


def _cmd_clicks_serve(args: argparse.Namespace) -> int:
    serve_click_ui(
        data_root=args.data_root,
        clicks_json=args.clicks_json,
        host=args.host,
        port=args.port,
    )
    return 0


def _cmd_clicks_status(args: argparse.Namespace) -> int:
    status = click_status(args.clicks_json)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


def _cmd_detect(args: argparse.Namespace) -> int:
    results = run_detection(
        data_root=args.data_root,
        out_root=args.out_root,
        clicks_json=args.clicks_json,
        only=args.only,
        text_prompt=args.text_prompt,
        save_masks=args.save_masks,
        overwrite=args.overwrite,
    )
    detected = sum(result.n_detected for result in results)
    frames = sum(result.n_frames for result in results)
    rate = detected / frames if frames else 0.0
    print(f"Detection complete: {len(results)} videos, {detected}/{frames} frames ({rate:.1%})")
    return 0


def _cmd_triangulate(args: argparse.Namespace) -> int:
    run_triangulation(
        raw_root=args.raw_root,
        formatted_root=args.formatted_root,
        detections_root=args.detections_root,
        out_root=args.out_root,
        runs=args.run,
        max_reproj_px=args.max_reproj_px,
        calibrate_only=args.calibrate_only,
        calibration_json=args.calibration_json,
    )
    return 0


def _cmd_log_summary(args: argparse.Namespace) -> int:
    summary = summarize_log_and_run(args.log, args.run_root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _cmd_calibrate_from_log(args: argparse.Namespace) -> int:
    result = calibrate_from_log(
        log_path=args.log,
        run_root=args.run_root,
        detections_root=args.detections_root,
        out_json=args.out_json,
        sample_count=args.sample_count,
        ransac_reproj_px=args.ransac_reproj_px,
        time_offset_s=args.time_offset_s,
        clock_lag_s=args.clock_lag_s,
        offset_search_radius_s=args.offset_search_radius_s,
        offset_search_step_s=args.offset_search_step_s,
        smoothing_m=args.smoothing_m,
        huber_px=args.huber_px,
        focal_sigma=args.focal_sigma,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_calibrate_from_mocap(args: argparse.Namespace) -> int:
    result = export_mocap_calibration(
        raw_root=args.raw_root,
        formatted_root=args.formatted_root,
        detections_root=args.detections_root,
        out_json=args.out_json,
        run=args.run,
        sample_count=args.sample_count,
        ransac_reproj_px=args.ransac_reproj_px,
        label_map=args.label_map,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_webui(args: argparse.Namespace) -> int:
    serve_webui(
        work_root=args.work_root,
        host=args.host,
        port=args.port,
    )
    return 0


def _cmd_run_all(args: argparse.Namespace) -> int:
    formatted_root = args.work_root / "formatted"
    detections_root = args.work_root / "detections"
    triangulation_root = args.work_root / "triangulation"

    if not args.skip_format:
        normalize_dataset(
            args.raw_root,
            formatted_root,
            only=args.only,
            overwrite=args.overwrite_format,
        )

    if not args.skip_detect:
        run_detection(
            data_root=formatted_root,
            out_root=detections_root,
            clicks_json=args.clicks_json,
            only=args.only,
            text_prompt=args.text_prompt,
            save_masks=args.save_masks,
            overwrite=args.overwrite_detect,
        )

    if not args.skip_triangulate:
        run_triangulation(
            raw_root=args.raw_root,
            formatted_root=formatted_root,
            detections_root=detections_root,
            out_root=triangulation_root,
            runs=args.run,
            max_reproj_px=args.max_reproj_px,
            calibration_json=args.calibration_json,
        )

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="firetrack",
        description="Docker-friendly Fire Tracking pipeline for the 5-27 multi-phone dataset.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fmt = subparsers.add_parser("format", help="Normalize raw 5-27 phone videos.")
    fmt.add_argument("--data-root", required=True, type=_path, help="Raw 5-27 dataset root.")
    fmt.add_argument("--out-root", required=True, type=_path, help="Normalized output root.")
    _add_only(fmt)
    fmt.add_argument("--overwrite", action="store_true", help="Recreate existing normalized videos.")
    fmt.set_defaults(func=_cmd_format)

    clicks = subparsers.add_parser("clicks", help="Prepare or inspect SAM click initialization.")
    click_sub = clicks.add_subparsers(dest="click_command", required=True)

    clicks_serve = click_sub.add_parser("serve", help="Serve browser UI for click annotation.")
    clicks_serve.add_argument("--data-root", required=True, type=_path, help="Normalized video root.")
    clicks_serve.add_argument("--clicks-json", required=True, type=_path, help="Click manifest path.")
    clicks_serve.add_argument("--host", default="0.0.0.0", help="Bind host.")
    clicks_serve.add_argument("--port", default=8080, type=int, help="Bind port.")
    clicks_serve.set_defaults(func=_cmd_clicks_serve)

    clicks_status = click_sub.add_parser("status", help="Print click manifest counts.")
    clicks_status.add_argument("--clicks-json", required=True, type=_path, help="Click manifest path.")
    clicks_status.set_defaults(func=_cmd_clicks_status)

    detect = subparsers.add_parser("detect", help="Run SAM3 2D video processing.")
    detect.add_argument("--data-root", required=True, type=_path, help="Normalized video root.")
    detect.add_argument("--out-root", required=True, type=_path, help="Detection output root.")
    detect.add_argument("--clicks-json", type=_path, help="Optional approved-click manifest.")
    _add_only(detect)
    detect.add_argument("--text-prompt", default="drone", help="SAM text prompt.")
    detect.add_argument("--save-masks", action="store_true", help="Also save compressed frame masks.")
    detect.add_argument("--overwrite", action="store_true", help="Recompute existing detections.")
    detect.set_defaults(func=_cmd_detect)

    tri = subparsers.add_parser("triangulate", help="Run mocap-calibrated triangulation.")
    tri.add_argument("--raw-root", required=True, type=_path, help="Raw 5-27 dataset root with mocap TSVs.")
    tri.add_argument("--formatted-root", required=True, type=_path, help="Normalized video root.")
    tri.add_argument("--detections-root", required=True, type=_path, help="Detection output root.")
    tri.add_argument("--out-root", required=True, type=_path, help="Triangulation output root.")
    tri.add_argument("--run", action="append", default=None, help="Run name to triangulate. May be repeated.")
    tri.add_argument("--max-reproj-px", default=30.0, type=float, help="Maximum per-view reprojection error.")
    tri.add_argument("--calibrate-only", action="store_true", help="Only estimate and save calibration.")
    tri.add_argument("--calibration-json", type=_path, help="Calibration JSON to read or write.")
    tri.set_defaults(func=_cmd_triangulate)

    log_summary = subparsers.add_parser("log-summary", help="Show ArduPilot log/camera-run timestamp overlap.")
    log_summary.add_argument("--log", required=True, type=_path, help="ArduPilot DataFlash .BIN log.")
    log_summary.add_argument("--run-root", required=True, type=_path, help="Camera run root with cam*/metadata.json.")
    log_summary.set_defaults(func=_cmd_log_summary)

    calib_log = subparsers.add_parser("calibrate-from-log", help="Estimate upload-mode camera extrinsics from a flight log.")
    calib_log.add_argument("--log", required=True, type=_path, help="ArduPilot DataFlash .BIN log.")
    calib_log.add_argument("--run-root", required=True, type=_path, help="Calibration camera run root with cam*/camera.json.")
    calib_log.add_argument("--detections-root", required=True, type=_path, help="Detection output root containing centroids.npz files.")
    calib_log.add_argument("--out-json", required=True, type=_path, help="Calibration JSON to write for upload triangulation.")
    calib_log.add_argument("--sample-count", default=600, type=int, help="Maximum detections per camera, including held-out validation.")
    calib_log.add_argument("--ransac-reproj-px", default=8.0, type=float, help="PnP RANSAC reprojection threshold.")
    calib_log.add_argument("--clock-lag-s", default=0.22, type=float, help="Initial shared camera_time - log_time lag in seconds.")
    calib_log.add_argument("--time-offset-s", default=None, type=float, help=argparse.SUPPRESS)
    calib_log.add_argument("--offset-search-radius-s", default=2.0, type=float, help="Shared-lag bounds +/- this many seconds around --clock-lag-s; zero fixes timing.")
    calib_log.add_argument("--offset-search-step-s", default=0.25, type=float, help="Maximum spacing between PnP initialization offsets; refinement is continuous.")
    calib_log.add_argument("--smoothing-m", default=0.1, type=float, help="Per-axis spline RMS smoothing allowance in meters; zero interpolates log positions.")
    calib_log.add_argument("--huber-px", default=3.0, type=float, help="Huber transition in pixels for each 2D detection residual.")
    calib_log.add_argument("--focal-sigma", default=float("inf"), type=float, help="0 fixes common fx=fy; inf estimates one common focal length per camera; e.g. 0.02 gives it a 2%% prior.")
    calib_log.set_defaults(func=_cmd_calibrate_from_log)

    calib_mocap = subparsers.add_parser("calibrate-from-mocap", help="Export upload-mode camera calibration from a mocap run.")
    calib_mocap.add_argument("--raw-root", required=True, type=_path, help="Raw 5-27 dataset root with mocap TSVs.")
    calib_mocap.add_argument("--formatted-root", required=True, type=_path, help="Normalized video root.")
    calib_mocap.add_argument("--detections-root", required=True, type=_path, help="Detection output root containing centroids.npz files.")
    calib_mocap.add_argument("--out-json", required=True, type=_path, help="Upload-mode calibration JSON to write.")
    calib_mocap.add_argument("--run", required=True, help="Mocap run name to calibrate from, e.g. ardu_run1.")
    calib_mocap.add_argument("--sample-count", default=160, type=int, help="Maximum correspondences per camera.")
    calib_mocap.add_argument("--ransac-reproj-px", default=8.0, type=float, help="PnP RANSAC reprojection threshold.")
    calib_mocap.add_argument(
        "--label-map",
        action="append",
        default=None,
        help="Map formatted/source labels to upload labels, e.g. camera1=cam1. May be repeated.",
    )
    calib_mocap.set_defaults(func=_cmd_calibrate_from_mocap)

    run_all = subparsers.add_parser("run-all", help="Run format, detect, and triangulate in sequence.")
    run_all.add_argument("--raw-root", required=True, type=_path, help="Raw 5-27 dataset root.")
    run_all.add_argument("--work-root", required=True, type=_path, help="Root for formatted/detections/triangulation.")
    run_all.add_argument("--clicks-json", type=_path, help="Optional approved-click manifest.")
    _add_only(run_all)
    run_all.add_argument("--run", action="append", default=None, help="Run name to triangulate. May be repeated.")
    run_all.add_argument("--text-prompt", default="drone", help="SAM text prompt.")
    run_all.add_argument("--save-masks", action="store_true", help="Also save compressed frame masks.")
    run_all.add_argument("--overwrite-format", action="store_true", help="Recreate existing normalized videos.")
    run_all.add_argument("--overwrite-detect", action="store_true", help="Recompute existing detections.")
    run_all.add_argument("--skip-format", action="store_true", help="Reuse existing formatted videos.")
    run_all.add_argument("--skip-detect", action="store_true", help="Reuse existing detections.")
    run_all.add_argument("--skip-triangulate", action="store_true", help="Skip triangulation.")
    run_all.add_argument("--max-reproj-px", default=30.0, type=float, help="Maximum per-view reprojection error.")
    run_all.add_argument("--calibration-json", type=_path, help="Calibration JSON to read or write.")
    run_all.set_defaults(func=_cmd_run_all)

    webui = subparsers.add_parser("webui", help="Serve the full-pipeline web dashboard.")
    webui.add_argument("--work-root", default="/work", type=_path, help="Root for formatted/detections/triangulation/uploads.")
    webui.add_argument("--host", default="0.0.0.0", help="Bind host.")
    webui.add_argument("--port", default=8080, type=int, help="Bind port.")
    webui.set_defaults(func=_cmd_webui)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
