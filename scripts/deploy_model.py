#!/usr/bin/env python
"""Start FastAPI inference server.
Usage: python scripts/deploy_model.py --model-path models/vector_field_mlp.pth
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Deploy SIRA inference API.")
    parser.add_argument('--model-path', default='models/vector_field_mlp.pth')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    os.environ['MODEL_PATH'] = args.model_path
    try:
        import uvicorn
        from src.inference.api.server import app
        print(f"Starting SIRA API server at http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
    except ImportError:
        print("uvicorn not installed. Install with: pip install uvicorn fastapi")


if __name__ == '__main__':
    main()
