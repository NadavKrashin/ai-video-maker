#!/usr/bin/env python3
"""Backwards-compatible entry point.

The implementation now lives in the ``ai_video_maker`` package. This thin shim
keeps `python pipeline.py ...` (and every command in the README) working. After
`pip install -e .` you can also use the `ai-video-maker` console command.
"""
from ai_video_maker.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
