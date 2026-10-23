# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包规范配置 - deploy_api
专为与 print-server 对接优化：
- 排除未使用的重型库（tkinter, matplotlib, torch 等）
- 自动收集 onnx 模型权重（hivision_modnet, rmbg-1.4, retinaface, mtcnn 等）
- 完整包含 uvicorn / fastapi 异步网络服务运行时
- 采用 onedir 目录模式，启动速度快（0.2秒秒开），避免 onefile 模式解压 400MB 带来的卡顿和临时垃圾
"""
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# 1. 收集 mtcnnruntime 权重
datas = collect_data_files('mtcnnruntime')

# 2. 收集 hivision 内部权重文件
weights_to_add = [
    (os.path.join('hivision', 'creator', 'weights', 'hivision_modnet.onnx'), os.path.join('hivision', 'creator', 'weights')),
    (os.path.join('hivision', 'creator', 'weights', 'modnet_photographic_portrait_matting.onnx'), os.path.join('hivision', 'creator', 'weights')),
    (os.path.join('hivision', 'creator', 'weights', 'rmbg-1.4.onnx'), os.path.join('hivision', 'creator', 'weights')),
    (os.path.join('hivision', 'creator', 'retinaface', 'weights', 'retinaface-resnet50.onnx'), os.path.join('hivision', 'creator', 'retinaface', 'weights')),
]

for src, dst in weights_to_add:
    if os.path.exists(src):
        datas.append((src, dst))

# 3. 收集隐式依赖
hiddenimports = [
    'uvicorn',
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.http.h11_impl',
    'uvicorn.protocols.http.httptools_impl',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'uvicorn.lifespan.off',
    'fastapi',
    'pydantic',
    'starlette',
    'starlette.middleware.cors',
    'onnxruntime',
    'mtcnnruntime',
    'cv2',
    'PIL',
    'numpy',
] + collect_submodules('uvicorn')

a = Analysis(
    ['deploy_api.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'scipy',
        'torch',
        'gradio',
        'pytest',
        'unittest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# 默认生成 onedir 模式（最适合集成到 Electron 安装包中，启动极快）
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='deploy_api',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='deploy_api',
)
