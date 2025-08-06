import os
import time
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import cv2
import numpy as np

# === 配置参数 ===
START_TIME = datetime(2025, 7, 1, 0, 0)
STEP_HOURS = 1
FRAMES = 24 * (31 + 5) + 16
OUTPUT_DIR = "nullschool_temp_finland"
GIF_NAME = "earth_temp_finland_ultra_hd.gif"
MP4_NAME = "earth_temp_finland_ultra_hd.mp4"
DELAY_PER_FRAME = 0.2  # 秒
LAT = 30.0
LON = 50.0
ZOOM = 1100  # 缩放值越大越近，200-600间

# 设置浏览器窗口大小（超清）
WINDOW_WIDTH = 3840
WINDOW_HEIGHT = 2160

# 字体设置
FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 72)

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 配置无头浏览器
options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument(f'--window-size={WINDOW_WIDTH},{WINDOW_HEIGHT}')
driver = webdriver.Chrome(options=options)

# 截图并处理
images = []
for i in tqdm(range(FRAMES), desc="Capturing frames"):
    t = START_TIME + timedelta(hours=i * STEP_HOURS)
    t_str = t.strftime('%Y-%m-%d-%H')
    timestamp_str = t.strftime('%Y-%m-%d %H:%M UTC')
    url_time = t.strftime('%Y/%m/%d/%H00Z')
    url = f"https://earth.nullschool.net/#{url_time}/wind/surface/level/overlay=temp/orthographic={LON:.2f},{LAT:.2f},{ZOOM}"

    driver.get(url)
    # time.sleep(1)

    path = f"{OUTPUT_DIR}/frame_{t_str}.png"
    driver.save_screenshot(path)

    img = Image.open(path)
    crop_width = int(WINDOW_WIDTH * 0.6)
    crop_height = int(WINDOW_HEIGHT * 1)
    left = (WINDOW_WIDTH - crop_width) // 2
    right = left + crop_width
    img_cropped = img.crop((left, 0, right, crop_height))

    # 添加时间戳文字到右上角（带描边）
    draw = ImageDraw.Draw(img_cropped)
    margin = 40
    bbox = draw.textbbox((0, 0), timestamp_str, font=FONT)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = img_cropped.width - text_w - margin
    y = margin

    # 白字 + 黑色描边效果
    outline_range = 2
    for dx in range(-outline_range, outline_range + 1):
        for dy in range(-outline_range, outline_range + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), timestamp_str, font=FONT, fill=(0, 0, 0, 255))
    draw.text((x, y), timestamp_str, font=FONT, fill=(255, 255, 255, 255))

    img_cropped.save(path)
    images.append(img_cropped)

driver.quit()

# 合成 GIF
images[0].save(
    GIF_NAME,
    save_all=True,
    append_images=images[1:],
    duration=int(DELAY_PER_FRAME * 1000),
    loop=0
)
print(f"✅ 聚焦北半球 GIF 已生成：{GIF_NAME}")

# 合成 MP4
frame_array = []
for img in images:
    frame = np.array(img.convert('RGB'))
    frame_array.append(frame)

height, width, _ = frame_array[0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(MP4_NAME, fourcc, 1 / DELAY_PER_FRAME, (width, height))

for frame in frame_array:
    video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

video.release()
print(f"✅ 聚焦北半球 MP4 已生成：{MP4_NAME}")
