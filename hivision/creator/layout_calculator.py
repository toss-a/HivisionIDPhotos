#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2024/9/5 21:35
@File: layout_calculator.py
@IDE: pycharm
@Description:
    布局计算器
"""

import cv2.detail
import numpy as np


def judge_layout(
    input_width,
    input_height,
    PHOTO_INTERVAL_W,
    PHOTO_INTERVAL_H,
    LIMIT_BLOCK_W,
    LIMIT_BLOCK_H,
    layout_direction=None,  # None for auto, 1 for horizontal, 2 for vertical
):
    centerBlockHeight_1, centerBlockWidth_1 = (
        input_height,
        input_width,
    )  # 由证件照们组成的一个中心区块（1 代表不转置排列）
    centerBlockHeight_2, centerBlockWidth_2 = (
        input_width,
        input_height,
    )  # 由证件照们组成的一个中心区块（2 代表转置排列）

    # 1.不转置排列的情况下：
    layout_col_no_transpose = 0  # 行
    layout_row_no_transpose = 0  # 列
    for i in range(1, 4):
        centerBlockHeight_temp = input_height * i + PHOTO_INTERVAL_H * (i - 1)
        if centerBlockHeight_temp < LIMIT_BLOCK_H:
            centerBlockHeight_1 = centerBlockHeight_temp
            layout_row_no_transpose = i
        else:
            break
    for j in range(1, 9):
        centerBlockWidth_temp = input_width * j + PHOTO_INTERVAL_W * (j - 1)
        if centerBlockWidth_temp < LIMIT_BLOCK_W:
            centerBlockWidth_1 = centerBlockWidth_temp
            layout_col_no_transpose = j
        else:
            break
    layout_number_no_transpose = layout_row_no_transpose * layout_col_no_transpose

    # 2.转置排列的情况下：
    layout_col_transpose = 0  # 行
    layout_row_transpose = 0  # 列
    for i in range(1, 4):
        centerBlockHeight_temp = input_width * i + PHOTO_INTERVAL_H * (i - 1)
        if centerBlockHeight_temp < LIMIT_BLOCK_H:
            centerBlockHeight_2 = centerBlockHeight_temp
            layout_row_transpose = i
        else:
            break
    for j in range(1, 9):
        centerBlockWidth_temp = input_height * j + PHOTO_INTERVAL_W * (j - 1)
        if centerBlockWidth_temp < LIMIT_BLOCK_W:
            centerBlockWidth_2 = centerBlockWidth_temp
            layout_col_transpose = j
        else:
            break
    layout_number_transpose = layout_row_transpose * layout_col_transpose

    # 如果用户指定了排版方向，则使用指定的方向
    if layout_direction == 1:  # 水平排版
        layout_mode = (layout_col_no_transpose, layout_row_no_transpose, 1)
        return layout_mode, centerBlockWidth_1, centerBlockHeight_1
    elif layout_direction == 2:  # 垂直排版
        layout_mode = (layout_col_transpose, layout_row_transpose, 2)
        return layout_mode, centerBlockWidth_2, centerBlockHeight_2
    else:  # 自动选择（默认行为）
        if layout_number_transpose > layout_number_no_transpose:
            layout_mode = (layout_col_transpose, layout_row_transpose, 2)
            return layout_mode, centerBlockWidth_2, centerBlockHeight_2
        else:
            layout_mode = (layout_col_no_transpose, layout_row_no_transpose, 1)
            return layout_mode, centerBlockWidth_1, centerBlockHeight_1


def generate_layout_array(input_height, input_width, LAYOUT_WIDTH=1795, LAYOUT_HEIGHT=1205, layout_direction=None):
    # 1.基础参数表
    PHOTO_INTERVAL_H = 30  # 证件照与证件照之间的垂直距离
    PHOTO_INTERVAL_W = 30  # 证件照与证件照之间的水平距离
    SIDES_INTERVAL_H = 50  # 证件照与画布边缘的垂直距离
    SIDES_INTERVAL_W = 70  # 证件照与画布边缘的水平距离
    LIMIT_BLOCK_W = LAYOUT_WIDTH - 2 * SIDES_INTERVAL_W
    LIMIT_BLOCK_H = LAYOUT_HEIGHT - 2 * SIDES_INTERVAL_H

    # 2.创建一个 1180x1746 的空白画布
    white_background = np.zeros([LAYOUT_HEIGHT, LAYOUT_WIDTH, 3], np.uint8)
    white_background.fill(255)

    # 3.计算照片的 layout（列、行、横竖朝向）,证件照组成的中心区块的分辨率
    layout_mode, centerBlockWidth, centerBlockHeight = judge_layout(
        input_width,
        input_height,
        PHOTO_INTERVAL_W,
        PHOTO_INTERVAL_H,
        LIMIT_BLOCK_W,
        LIMIT_BLOCK_H,
        layout_direction,
    )
    # 4.开始排列组合
    x11 = (LAYOUT_WIDTH - centerBlockWidth) // 2
    y11 = (LAYOUT_HEIGHT - centerBlockHeight) // 2
    typography_arr = []
    typography_rotate = False
    if layout_mode[2] == 2:
        input_height, input_width = input_width, input_height
        typography_rotate = True

    for j in range(layout_mode[1]):
        for i in range(layout_mode[0]):
            xi = x11 + i * input_width + i * PHOTO_INTERVAL_W
            yi = y11 + j * input_height + j * PHOTO_INTERVAL_H
            typography_arr.append([xi, yi])

    return typography_arr, typography_rotate


def generate_layout_image(
    input_image, typography_arr, typography_rotate, width=295, height=413, 
    crop_line:bool=False,
    LAYOUT_WIDTH=1795, 
    LAYOUT_HEIGHT=1205,
):
  
    # 创建一个白色背景的空白画布
    white_background = np.zeros([LAYOUT_HEIGHT, LAYOUT_WIDTH, 3], np.uint8)
    white_background.fill(255)
    
    # 如果输入图像的高度不等于指定高度，则调整图像大小
    if input_image.shape[0] != height:
        input_image = cv2.resize(input_image, (width, height))
    
    # 如果需要旋转排版，则对图像进行转置和垂直镜像
    if typography_rotate:
        input_image = cv2.transpose(input_image)
        input_image = cv2.flip(input_image, 0)  # 0 表示垂直镜像

        # 交换高度和宽度
        height, width = width, height
    
    # 将图像按照排版数组中的位置放置到白色背景上
    for arr in typography_arr:
        locate_x, locate_y = arr[0], arr[1]
        white_background[locate_y : locate_y + height, locate_x : locate_x + width] = (
            input_image
        )

    if crop_line:
        # 添加裁剪线 - 在每张照片边缘绘制灰色线条
        line_color = (200, 200, 200)  # 浅灰色
        line_thickness = 1

        # 为每张照片绘制边框线条
        for arr in typography_arr:
            x, y = arr[0], arr[1]

            # 绘制照片的四条边缘线
            # 左边缘
            cv2.line(white_background, (x, y), (x, y + height), line_color, line_thickness)
            # 右边缘
            cv2.line(white_background, (x + width, y), (x + width, y + height), line_color, line_thickness)
            # 上边缘
            cv2.line(white_background, (x, y), (x + width, y), line_color, line_thickness)
            # 下边缘
            cv2.line(white_background, (x, y + height), (x + width, y + height), line_color, line_thickness)

    # 返回排版后的图像
    return white_background


def generate_mixed_layout_image(
    image_1_inch, image_2_inch,
    crop_line: bool = False,
    LAYOUT_WIDTH=1795,
    LAYOUT_HEIGHT=1205,
):
    """
    混合排版：在6寸纸张上排4张一寸照片和2张两寸照片
    一寸照片尺寸：295x413 (宽x高)
    两寸照片尺寸：413x626 (宽x高)

    布局方案：左侧4张一寸（2x2），右侧2张两寸（竖排）
    """
    # 一寸和两寸照片的标准尺寸
    width_1_inch, height_1_inch = 295, 413
    width_2_inch, height_2_inch = 413, 626

    PHOTO_INTERVAL = 30

    # 调整图像大小
    if image_1_inch.shape[:2] != (height_1_inch, width_1_inch):
        image_1_inch = cv2.resize(image_1_inch, (width_1_inch, height_1_inch))
    if image_2_inch.shape[:2] != (height_2_inch, width_2_inch):
        image_2_inch = cv2.resize(image_2_inch, (width_2_inch, height_2_inch))

    # 创建白色背景
    white_background = np.zeros([LAYOUT_HEIGHT, LAYOUT_WIDTH, 3], np.uint8)
    white_background.fill(255)

    # 计算布局：左侧4张一寸（2x2），右侧2张两寸（竖排）
    # 左侧一寸照片区域
    left_section_width = width_1_inch * 2 + PHOTO_INTERVAL
    left_section_height = height_1_inch * 2 + PHOTO_INTERVAL

    # 右侧两寸照片区域（竖排，但只放1张，因为2张会超高）
    # 实际上2张两寸竖排需要 626*2+30=1282 > 1205，所以改为横向旋转放置
    # 或者缩小间距，让我们尝试紧凑布局

    # 方案：左侧4张一寸（2x2），右侧2张两寸横向放置（旋转90度）
    # 两寸照片旋转后：626x413 变成 413x626
    right_section_width = height_2_inch  # 旋转后的宽度
    right_section_height = width_2_inch * 2 + PHOTO_INTERVAL  # 旋转后2张的高度

    # 总宽度
    total_width = left_section_width + PHOTO_INTERVAL + right_section_width

    # 使用较大的高度
    max_height = max(left_section_height, right_section_height)

    # 计算起始位置（居中）
    start_x = (LAYOUT_WIDTH - total_width) // 2
    start_y = (LAYOUT_HEIGHT - max_height) // 2

    # 放置4张一寸照片（左侧，2x2排列）
    positions_1_inch = []
    for row in range(2):
        for col in range(2):
            x = start_x + col * (width_1_inch + PHOTO_INTERVAL)
            y = start_y + row * (height_1_inch + PHOTO_INTERVAL)
            white_background[y:y + height_1_inch, x:x + width_1_inch] = image_1_inch
            positions_1_inch.append((x, y, width_1_inch, height_1_inch))

    # 放置2张两寸照片（右侧，横向旋转90度后竖排）
    positions_2_inch = []
    right_start_x = start_x + left_section_width + PHOTO_INTERVAL

    # 旋转两寸照片（逆时针90度）
    image_2_inch_rotated = cv2.rotate(image_2_inch, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rotated_height, rotated_width = image_2_inch_rotated.shape[:2]  # 应该是 413x626

    for i in range(2):
        x = right_start_x
        y = start_y + i * (rotated_height + PHOTO_INTERVAL)
        white_background[y:y + rotated_height, x:x + rotated_width] = image_2_inch_rotated
        positions_2_inch.append((x, y, rotated_width, rotated_height))

    # 添加裁剪线 - 在每张照片边缘绘制灰色线条
    if crop_line:
        line_color = (200, 200, 200)  # 浅灰色
        line_thickness = 1

        # 为所有照片添加边框线条
        for x, y, w, h in positions_1_inch + positions_2_inch:
            # 左边缘
            cv2.line(white_background, (x, y), (x, y + h), line_color, line_thickness)
            # 右边缘
            cv2.line(white_background, (x + w, y), (x + w, y + h), line_color, line_thickness)
            # 上边缘
            cv2.line(white_background, (x, y), (x + w, y), line_color, line_thickness)
            # 下边缘
            cv2.line(white_background, (x, y + h), (x + w, y + h), line_color, line_thickness)

    return white_background
