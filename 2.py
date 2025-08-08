import os
import time
import math
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import cv2
import numpy as np
import requests

class EarthNullschoolAnimator:
    def __init__(self, start_time, step_hours, num_frames,
                 center_lat, center_lon, zoom,
                 window_width, window_height,
                 output_dir, gif_name, mp4_name,
                 delay_per_frame, font_path, font_size):

        self.start_time = start_time
        self.step_hours = step_hours
        self.num_frames = num_frames
        self.lat = center_lat
        self.lon = center_lon
        self.zoom = zoom
        self.window_width = window_width
        self.window_height = window_height
        self.output_dir = output_dir
        self.gif_name = gif_name
        self.mp4_name = mp4_name
        self.delay_per_frame = delay_per_frame

        self.font = ImageFont.truetype(font_path, font_size)
        self.images = []

        os.makedirs(self.output_dir, exist_ok=True)
        self.driver = self._init_driver()

        # 不再使用硬编码像素坐标；仅保留经纬度
        self.cities = [
            {"name": "Beijing",  "lat": 39.9, "lon": 116.4},
            {"name": "Shanghai", "lat": 31.2, "lon": 121.5},
            {"name": "Harbin",   "lat": 45.8, "lon": 126.5},
            {"name": "Urumqi",   "lat": 43.8, "lon": 87.6},
        ]

    # ---------- 浏览器 ----------
    def _init_driver(self):
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument(f'--window-size={self.window_width},{self.window_height}')
        return webdriver.Chrome(options=options)

    # ---------- URL ----------
    def _build_url(self, timestamp):
        url_time = timestamp.strftime('%Y/%m/%d/%H00Z')
        return (
            f"https://earth.nullschool.net/#{url_time}"
            f"/wind/surface/level/overlay=temp"
            f"/orthographic={self.lon:.2f},{self.lat:.2f},{self.zoom}"
        )

    # ---------- 数据 ----------
    def _get_temperature(self, lat, lon, timestamp):
        date_str = timestamp.strftime('%Y-%m-%dT%H:00')
        url = (
            f"https://api.open-meteo.com/v1/gfs?latitude={lat}&longitude={lon}"
            f"&hourly=temperature_2m&timezone=UTC"
            f"&start_date={timestamp.strftime('%Y-%m-%d')}"
            f"&end_date={timestamp.strftime('%Y-%m-%d')}"
        )
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()
            times = data['hourly']['time']
            temps = data['hourly']['temperature_2m']
            for i, t in enumerate(times):
                if t.startswith(date_str):
                    return temps[i], f"{temps[i]:.1f}°C"
        except Exception as e:
            print(f"⚠️ 获取温度失败：{e}")
        return None, "N/A"

    # ---------- 投影：经纬度 -> 像素 ----------
    def _geo_to_pixel_orthographic(self, lat_city, lon_city, img_w, img_h):
        """
        将经纬度映射到正交投影图像像素坐标（以裁剪后的图像为基准）。
        若点在地球背面，返回 None。
        """
        R = min(img_w, img_h) * 0.45  # 屏幕上地球半径（像素），留一些边距
        lat0 = math.radians(self.lat)
        lon0 = math.radians(self.lon)
        lat = math.radians(lat_city)
        lon = math.radians(lon_city)

        cos_c = math.sin(lat0) * math.sin(lat) + math.cos(lat0) * math.cos(lat) * math.cos(lon - lon0)
        if cos_c < 0:
            return None  # 背面不可见

        x = R * math.cos(lat) * math.sin(lon - lon0)
        y = R * (math.cos(lat0) * math.sin(lat) - math.sin(lat0) * math.cos(lat) * math.cos(lon - lon0))

        px = int(img_w / 2 + x)
        py = int(img_h / 2 - y)
        return (px, py)

    # ---------- 帧捕获 ----------
    def _capture_frame(self, i):
        t = self.start_time + timedelta(hours=i * self.step_hours)
        t_str = t.strftime('%Y-%m-%d-%H')
        timestamp_str = t.strftime('%Y-%m-%d %H:%M UTC')

        url = self._build_url(t)
        self.driver.get(url)
        time.sleep(1)  # 简单等待渲染完成

        path = os.path.join(self.output_dir, f"frame_{t_str}.png")
        self.driver.save_screenshot(path)

        img = Image.open(path)
        img_cropped = self._crop_view(img)
        self._draw_timestamp(img_cropped, timestamp_str)
        self._draw_overlays(img_cropped, t)

        img_cropped.save(path)
        self.images.append(img_cropped)

    # ---------- 裁剪 ----------
    def _crop_view(self, img):
        crop_width = int(self.window_width * 0.6)
        crop_height = int(self.window_height * 1.0)
        left = (self.window_width - crop_width) // 2
        right = left + crop_width
        return img.crop((left, 0, right, crop_height))

    # ---------- 绘制：时间戳 ----------
    def _draw_timestamp(self, img, text):
        draw = ImageDraw.Draw(img)
        margin = 40
        bbox = draw.textbbox((0, 0), text, font=self.font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = img.width - text_w - margin
        y = margin
        # 描边
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx or dy:
                    draw.text((x + dx, y + dy), text, font=self.font, fill=(0, 0, 0))
        draw.text((x, y), text, font=self.font, fill=(255, 255, 255))

    # ---------- 绘制：城市叠加 ----------
    def _draw_overlays(self, img, timestamp):
        draw = ImageDraw.Draw(img)
        for city in self.cities:
            temp_val, temp_label = self._get_temperature(city['lat'], city['lon'], timestamp)
            label = f"{city['name']} {temp_label}"

            xy = self._geo_to_pixel_orthographic(city['lat'], city['lon'], img.width, img.height)
            if xy is None:
                continue  # 不在可见面

            x, y = xy
            color = self._temp_to_color(temp_val) if temp_val is not None else (255, 255, 255)
            r = 12
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

            bbox = draw.textbbox((0, 0), label, font=self.font)
            text_h = bbox[3] - bbox[1]
            tx, ty = x + r + 6, y - text_h // 2
            # 描边
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx or dy:
                        draw.text((tx + dx, ty + dy), label, font=self.font, fill=(0, 0, 0))
            draw.text((tx, ty), label, font=self.font, fill=(255, 255, 255))

    # ---------- 颜色映射 ----------
    def _temp_to_color(self, temp):
        # 简单色标：<0 冷蓝，0~25 过渡到黄，>25 过渡到红
        if temp < 0:
            return (0, 160, 255)
        elif temp < 25:
            g = int(255 * (temp / 25))
            return (255 - g, 255, 0)
        else:
            r = min(255, int(255 * (temp - 25) / 15))
            return (255, 255 - r, 0)

    # ---------- 主流程 ----------
    def run_capture(self):
        for i in tqdm(range(self.num_frames), desc="Capturing frames"):
            self._capture_frame(i)
        self.driver.quit()

    def export_gif(self):
        gif_path = os.path.join(self.output_dir, self.gif_name)
        self.images[0].save(
            gif_path,
            save_all=True,
            append_images=self.images[1:],
            duration=int(self.delay_per_frame * 1000),
            loop=0
        )
        print(f"✅ GIF 已生成：{gif_path}")

    def export_mp4(self):
        mp4_path = os.path.join(self.output_dir, self.mp4_name)
        frame_array = [np.array(img.convert('RGB')) for img in self.images]
        height, width, _ = frame_array[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(mp4_path, fourcc, 1 / self.delay_per_frame, (width, height))
        for frame in frame_array:
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video.release()
        print(f"✅ MP4 已生成：{mp4_path}")


if __name__ == '__main__':
    animator = EarthNullschoolAnimator(
        start_time=datetime(2025, 7, 1, 0, 0),
        step_hours=1,
        num_frames=3,  # 调试：先跑三帧
        center_lat=35.0,
        center_lon=105.0,
        zoom=1100,
        window_width=3840,
        window_height=2160,
        output_dir="nullschool_temp_china",
        gif_name="earth_temp_china_ultra_hd.gif",
        mp4_name="earth_temp_china_ultra_hd.mp4",
        delay_per_frame=0.2,
        font_path="/System/Library/Fonts/Supplemental/Arial.ttf",
        font_size=72
    )

    animator.run_capture()
    animator.export_gif()
    animator.export_mp4()
