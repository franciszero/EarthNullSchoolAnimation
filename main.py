import os
import time
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image
import imageio
from tqdm import tqdm

# === 配置参数 ===
START_TIME = datetime(2025, 7, 1, 0, 0)
STEP_HOURS = 1
FRAMES = 3  # * (31 + 5)
OUTPUT_DIR = "nullschool_temp_finland"
GIF_NAME = "earth_temp_finland.gif"
DELAY_PER_FRAME = 0.4  # 秒
LAT = 50.0
LON = 40.0
ZOOM = 400  # 越大越近，区域越小

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 启动无头浏览器
options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--window-size=1200,800')
driver = webdriver.Chrome(options=options)

# 截图循环
images = []
for i in tqdm(range(FRAMES), desc="Capturing frames"):
    t = START_TIME + timedelta(hours=i * STEP_HOURS)
    t_str = t.strftime('%Y-%m-%d-%H')
    url_time = t.strftime('%Y/%m/%d/%H00Z')
    url = f"https://earth.nullschool.net/#{url_time}/wind/surface/level/overlay=temp/orthographic={LON:.2f},{LAT:.2f},{ZOOM}"

    driver.get(url)
    time.sleep(5)

    path = f"{OUTPUT_DIR}/frame_{t_str}.png"
    driver.save_screenshot(path)

    img = Image.open(path)
    img_cropped = img.crop((200, 100, 1000, 700))  # 可调整裁剪尺寸
    img_cropped.save(path)  # 可选保存裁剪图
    images.append(img_cropped)

driver.quit()

# 合成 GIF
images[0].save(GIF_NAME, save_all=True, append_images=images[1:], duration=int(DELAY_PER_FRAME * 1000), loop=0)
print(f"✅ 动图已保存为：{GIF_NAME}")
