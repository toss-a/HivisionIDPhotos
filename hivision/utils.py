#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
精简后的通用图像与数据处理工具
"""
import io
import os
import sys
import base64
import numpy as np
import cv2
from PIL import Image


def get_resource_path(*relative_parts: str) -> str:
    """
    智能获取资源/模型文件的绝对路径，自动适配常规源码运行、PyInstaller 单文件/单目录打包：
    1. 优先尝试 PyInstaller 运行时解压/内部目录 (sys._MEIPASS)
    2. 尝试可执行文件所在目录 (sys.executable 所在目录)
    3. 尝试源码所在 hivision 根目录及当前工作目录
    """
    rel_path = os.path.join(*relative_parts)

    candidate_roots = []

    # 1. PyInstaller 运行时目录
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            candidate_roots.append(sys._MEIPASS)
        candidate_roots.append(os.path.dirname(sys.executable))

    # 2. 源码所在 hivision 根目录及工作目录
    here = os.path.dirname(os.path.abspath(__file__))  # hivision/
    repo_root = os.path.dirname(here)                 # repo root/
    candidate_roots.extend([repo_root, here, os.getcwd()])

    for root in candidate_roots:
        target = os.path.join(root, rel_path)
        if os.path.exists(target):
            return target

    # 如果未找到，默认返回首选候选路径
    base = candidate_roots[0] if candidate_roots else ""
    return os.path.join(base, rel_path)


def save_image_dpi_to_bytes(image: np.ndarray, output_image_path: str = None, dpi: int = 300) -> bytes:
    """
    将 OpenCV (BGR/BGRA) 图像数组正确转换并保存为指定 DPI 的 PNG 字节流。
    """
    if len(image.shape) == 3 and image.shape[2] == 4:
        # OpenCV 输出为 BGRA，PIL 期望 RGBA，必须转换通道避免人物皮肤偏蓝
        rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        pil_img = Image.fromarray(rgba, mode="RGBA")
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # OpenCV 输出为 BGR，PIL 期望 RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb, mode="RGB")
    else:
        pil_img = Image.fromarray(image)

    byte_stream = io.BytesIO()
    pil_img.save(byte_stream, format="PNG", dpi=(dpi, dpi))
    image_bytes = byte_stream.getvalue()

    if output_image_path:
        with open(output_image_path, "wb") as f:
            f.write(image_bytes)

    return image_bytes


def save_image_dpi_to_file(image: np.ndarray, output_image_path: str, dpi: int = 300):
    """
    极速文件直接写入：将 OpenCV (BGR/BGRA) 图像以指定 DPI 保存到目标路径为标准 PNG (RGBA/RGB)
    """
    if len(image.shape) == 3 and image.shape[2] == 4:
        # OpenCV 输出为 BGRA，PIL 期望 RGBA，必须转换通道避免人物皮肤偏蓝
        rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        pil_img = Image.fromarray(rgba, mode="RGBA")
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # OpenCV 输出为 BGR，PIL 期望 RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb, mode="RGB")
    else:
        pil_img = Image.fromarray(image)

    pil_img.save(output_image_path, format="PNG", dpi=(dpi, dpi))


def bytes_2_base64(img_byte_arr: bytes) -> str:
    """字节流转 base64 字符串"""
    base64_image = base64.b64encode(img_byte_arr).decode("utf-8")
    return "data:image/png;base64," + base64_image


def base64_2_numpy(base64_image: str) -> np.ndarray:
    """base64 字符串解码为 numpy 图像"""
    if base64_image.startswith("data:image"):
        base64_image = base64_image.split(",")[1]
    img_bytes = base64.b64decode(base64_image)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)


def get_optimal_cpu_threads(max_cap: int = 8) -> int:
    """
    自动获取最适合 CPU ONNX 推理的线程数：
    1. 自动检测并优先使用物理核心数，规避超线程/SMT调度冲突与低压CPU功耗墙惩罚
    2. 对单图推理设置线程上限（默认8线程），避免多线程汇聚同步与缓存颠簸
    3. 支持环境变量 HIVISION_NUM_THREADS 手动覆盖
    4. 自动感知 Linux Docker / cgroups 容器配额
    """
    # 1. 允许通过环境变量强制指定
    env_threads = os.getenv("HIVISION_NUM_THREADS")
    if env_threads:
        try:
            return max(1, int(env_threads))
        except ValueError:
            pass

    physical_cores = None
    logical_cores = os.cpu_count() or 1

    # 2. 检查 Docker / cgroups CPU 配额限制（针对 Linux 容器部署）
    quota_cores = None
    if sys.platform.startswith("linux"):
        cgroup_v2 = "/sys/fs/cgroup/cpu.max"
        cgroup_v1_quota = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
        cgroup_v1_period = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
        try:
            if os.path.exists(cgroup_v2):
                with open(cgroup_v2, "r") as f:
                    quota, period = f.read().strip().split()
                    if quota != "max":
                        quota_cores = int(quota) / int(period)
            elif os.path.exists(cgroup_v1_quota) and os.path.exists(cgroup_v1_period):
                with open(cgroup_v1_quota, "r") as f_q, open(cgroup_v1_period, "r") as f_p:
                    quota = int(f_q.read().strip())
                    period = int(f_p.read().strip())
                    if quota > 0 and period > 0:
                        quota_cores = quota / period
        except Exception:
            pass

    # 3. 优先尝试 psutil 精准获取物理核心数
    try:
        import psutil
        physical_cores = psutil.cpu_count(logical=False)
    except Exception:
        pass

    # 4. Windows 纯标准库 ctypes 获取物理核心数（零子进程开销）
    if physical_cores is None and sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class _ProcessorCore(ctypes.Structure):
                _fields_ = [("Flags", wintypes.BYTE)]

            class _DummyUnion(ctypes.Union):
                _fields_ = [
                    ("ProcessorCore", _ProcessorCore),
                    ("NodeNumber", wintypes.DWORD),
                    ("Reserved", ctypes.c_ulonglong * 2),
                ]

            class SYSTEM_LOGICAL_PROCESSOR_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("ProcessorMask", ctypes.c_size_t),
                    ("Relationship", wintypes.DWORD),
                    ("u", _DummyUnion),
                ]

            kernel32 = ctypes.windll.kernel32
            length = wintypes.DWORD(0)
            kernel32.GetLogicalProcessorInformation(None, ctypes.byref(length))
            count = length.value // ctypes.sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)
            if count > 0:
                buffer = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION * count)()
                if kernel32.GetLogicalProcessorInformation(buffer, ctypes.byref(length)):
                    # RelationProcessorCore == 0
                    core_count = sum(1 for item in buffer if item.Relationship == 0)
                    if core_count > 0:
                        physical_cores = core_count
        except Exception:
            pass

    # 5. Linux 下通过 /proc/cpuinfo 统计独立物理核心
    if physical_cores is None and sys.platform.startswith("linux"):
        try:
            cores = set()
            with open("/proc/cpuinfo", "r") as f:
                core_id, phys_id = None, None
                for line in f:
                    if line.startswith("physical id"):
                        phys_id = line.split(":")[1].strip()
                    elif line.startswith("core id"):
                        core_id = line.split(":")[1].strip()
                    if phys_id is not None and core_id is not None:
                        cores.add((phys_id, core_id))
                        core_id, phys_id = None, None
            if cores:
                physical_cores = len(cores)
        except Exception:
            pass

    # 6. macOS 探测
    if physical_cores is None and sys.platform == "darwin":
        try:
            import subprocess
            out = subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"]).decode().strip()
            if out.isdigit():
                physical_cores = int(out)
        except Exception:
            pass

    # 7. 启发式安全兜底：逻辑核 >= 4 时通常存在超线程，折半取物理核；否则使用全部
    if physical_cores is None:
        if logical_cores >= 4:
            physical_cores = max(2, logical_cores // 2)
        else:
            physical_cores = logical_cores

    chosen = physical_cores

    # 8. 结合容器配额约束
    if quota_cores is not None:
        chosen = min(chosen, max(1, int(quota_cores)))

    # 9. 施加单图推理收益上限，防止轻薄本撞功耗墙与多线程竞争
    optimal = max(1, min(chosen, max_cap))
    return optimal


def get_onnx_session_options(intra_threads: int = None):
    """
    生成高度优化的 ONNX Runtime SessionOptions：
    - 自动开启全图优化（ORT_ENABLE_ALL）
    - 绑定最优物理线程数，规避超线程惩罚
    - 顺序执行降低切换开销
    """
    import onnxruntime
    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

    if intra_threads is None:
        intra_threads = get_optimal_cpu_threads()

    opts.intra_op_num_threads = intra_threads
    opts.inter_op_num_threads = 1
    return opts

