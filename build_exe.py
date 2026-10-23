#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键打包脚本：将 deploy_api 打包为 Windows 可执行程序 (EXE)
使用方式：
    python build_exe.py              # 默认构建 onedir 目录模式（推荐：启动极快，与 print-server 完美配合）
    python build_exe.py --onefile    # 构建单文件 exe 模式（产物为单个大 exe）
"""
import os
import sys
import shutil
import argparse
import subprocess

def run_build(onefile: bool = False):
    print("=" * 60)
    mode_str = "单文件 (--onefile)" if onefile else "目录模式 (--onedir, 推荐)"
    print(f"开始打包 deploy_api: 模式 = {mode_str}")
    print("=" * 60)

    # 1. 检查 PyInstaller
    try:
        import PyInstaller
        print(f"检测到 PyInstaller 版本: {PyInstaller.__version__}")
    except ImportError:
        print("错误: 未安装 PyInstaller，正在通过 uv/pip 自动安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # 2. 清理历史构建
    for d in ["build"]:
        if os.path.exists(d):
            print(f"清理目录: {d}")
            shutil.rmtree(d, ignore_errors=True)

    # 3. 构造打包命令
    if onefile:
        # 单文件打包命令
        cmd = [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--onefile",
            "--name", "deploy_api",
            "--collect-data", "mtcnnruntime",
            "--collect-submodules", "uvicorn",
            "--exclude-module", "matplotlib",
            "--exclude-module", "tkinter",
            "--exclude-module", "torch",
            "--exclude-module", "scipy",
            "--exclude-module", "gradio",
            # 添加权重
            "--add-data", f"hivision/creator/weights/face_detection_yunet_2023mar.onnx{os.pathsep}hivision/creator/weights",
            "--add-data", f"hivision/creator/weights/hivision_modnet.onnx{os.pathsep}hivision/creator/weights",
            "--add-data", f"hivision/creator/weights/rmbg-1.4.onnx{os.pathsep}hivision/creator/weights",
            "--add-data", f"hivision/creator/retinaface/weights/retinaface-resnet50.onnx{os.pathsep}hivision/creator/retinaface/weights",
            "deploy_api.py",
        ]
        portrait_weight = os.path.join("hivision", "creator", "weights", "modnet_photographic_portrait_matting.onnx")
        if os.path.exists(portrait_weight):
            cmd.insert(-1, "--add-data")
            cmd.insert(-1, f"{portrait_weight}{os.pathsep}hivision/creator/weights")
    else:
        # 使用配置好的 deploy_api.spec
        cmd = [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "deploy_api.spec",
        ]

    print("执行命令:\n", " ".join(cmd))
    ret = subprocess.call(cmd)
    if ret != 0:
        print(f"\n[FAIL] 打包失败，退出码: {ret}")
        sys.exit(ret)

    print("\n" + "=" * 60)
    print("[SUCCESS] 打包成功！产物位置:")
    if onefile:
        exe_path = os.path.abspath(os.path.join("dist", "deploy_api.exe"))
        print(f"  EXE 文件: {exe_path}")
    else:
        dist_dir = os.path.abspath(os.path.join("dist", "deploy_api"))
        exe_path = os.path.join(dist_dir, "deploy_api.exe")
        print(f"  目录路径: {dist_dir}")
        print(f"  主程序:   {exe_path}")
        print("\n提示: 将 dist/deploy_api 目录放置到 print-server 下，即可免 Python 环境直接运行！")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HivisionIDPhotos EXE 打包工具")
    parser.add_argument("--onefile", action="store_true", help="构建为独立单个 exe 文件 (默认构建为目录模式)")
    args = parser.parse_args()

    run_build(onefile=args.onefile)
