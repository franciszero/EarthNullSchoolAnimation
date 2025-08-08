import os
import time
import math
import json
from io import BytesIO
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
                 delay_per_frame, font_path, font_size,
                 cities_path: str | None = None,
                 lang: str = "en",
                 show_temp: bool = True,
                 show_wind: bool = True,
                 show_rh: bool = True,
                 show_pres: bool = False):

        self.start_time = start_time
        self.step_hours = step_hours
        self.num_frames = num_frames
        self.lat = center_lat
        self.lon = center_lon
        self.zoom = float(zoom)  # 用作正交投影像素半径
        self.window_width = window_width
        self.window_height = window_height
        self.output_dir = output_dir
        self.gif_name = gif_name
        self.mp4_name = mp4_name
        self.delay_per_frame = delay_per_frame

        # 显示选项
        self.lang = lang.lower()
        self.show_temp = show_temp
        self.show_wind = show_wind
        self.show_rh = show_rh
        self.show_pres = show_pres

        # 字体（以传入的 font_size 为基准，不再按窗口宽度重算）
        self.font_ts, self.font_lbl = self._init_fonts(font_path, font_size, self.lang)
        print("Font used ->", getattr(self, "_font_used", "unknown"))

        self.images = []

        os.makedirs(self.output_dir, exist_ok=True)
        self.driver = self._init_driver()

        # 城市：支持外部 JSON；否则使用示例
        self.cities = self._load_cities(cities_path) if cities_path else [
            {"name": "Beijing" if self.lang != "zh" else "北京", "lat": 39.9, "lon": 116.4},
            {"name": "Shanghai" if self.lang != "zh" else "上海", "lat": 31.2, "lon": 121.5},
            {"name": "Harbin" if self.lang != "zh" else "哈尔滨", "lat": 45.8, "lon": 126.5},
            {"name": "Urumqi" if self.lang != "zh" else "乌鲁木齐", "lat": 43.8, "lon": 87.6},
        ]

        # 按天缓存：key=(date, round(lat,2), round(lon,2)) -> dict of arrays
        self._met_cache: dict[tuple, dict] = {}

    # ---------- 字体 ----------
    def _init_fonts(self, preferred_path: str, base_size: int, lang: str):
        """
        用入参 base_size 作为时间戳字号，标签字号 = 0.85 * base_size。
        选择一个能覆盖中英+符号(° % | / -)的字体；TTC 尝试多个 index。
        """
        want = "风湿度°%|/-"  # 覆盖我们会用到的字形
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # 优先：覆盖广
            "/System/Library/Fonts/PingFang.ttc",                   # 中文系统字体
            preferred_path,                                         # 用户传入
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Verdana.ttf",
        ]

        def can_render(path, size, index=0):
            try:
                f = ImageFont.truetype(path, size, index=index)
                total = 0
                for ch in want:
                    if hasattr(f, "getlength"):
                        total += f.getlength(ch)
                    else:
                        total += f.getbbox(ch)[2]  # 旧 pillow 兼容
                return total > 0
            except Exception:
                return False

        chosen, idx = None, 0
        for p in candidates:
            if p.endswith(".ttc"):
                for i in range(0, 8):  # 简单尝试前几个字面
                    if can_render(p, base_size, i):
                        chosen, idx = p, i
                        break
            else:
                if can_render(p, base_size, 0):
                    chosen, idx = p, 0
            if chosen:
                break

        if not chosen:
            f = ImageFont.load_default()
            self._font_used = "PIL default"
            return f, f

        self._font_used = f"{chosen} (index {idx})"
        font_ts = ImageFont.truetype(chosen, max(14, int(base_size * 1.00)), index=idx)
        font_lbl = ImageFont.truetype(chosen, max(12, int(base_size * 0.85)), index=idx)
        return font_ts, font_lbl

    # ---------- 浏览器 ----------
    def _init_driver(self):
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument(f'--window-size={self.window_width},{self.window_height}')
        return webdriver.Chrome(options=options)

    # ---------- 城市加载 ----------
    @staticmethod
    def _load_cities(path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        norm = []
        for c in data:
            norm.append({
                "name": c.get("name") or c.get("city") or "",
                "lat": float(c["lat"]),
                "lon": float(c["lon"]),
            })
        return norm

    # ---------- URL ----------
    def _build_url(self, timestamp):
        url_time = timestamp.strftime('%Y/%m/%d/%H00Z')
        return (
            f"https://earth.nullschool.net/#{url_time}"
            f"/wind/surface/level/overlay=temp"
            f"/orthographic={self.lon:.2f},{self.lat:.2f},{self.zoom:.0f}"
        )

    # ---------- 页面等待（自适应） ----------
    def _wait_canvas(self, max_wait=3.0, step=0.25):
        waited = 0.0
        while waited < max_wait:
            try:
                state = self.driver.execute_script("return document.readyState")
                if state == "complete":
                    png = self.driver.get_screenshot_as_png()
                    gray = np.array(Image.open(BytesIO(png)).convert('L'))
                    if gray.mean() > 3:   # 阈值略放宽，适应更暗底图
                        return
            except Exception:
                pass
            time.sleep(step)
            waited += step

    # ---------- 气象数据（按天缓存，多要素） ----------
    def _get_met_at(self, lat, lon, timestamp):
        date_key = timestamp.strftime('%Y-%m-%d')
        lat2 = round(float(lat), 2)
        lon2 = round(float(lon), 2)
        key = (date_key, lat2, lon2)

        if key not in self._met_cache:
            params = [
                "temperature_2m",
                "wind_speed_10m", "wind_direction_10m",
                "relative_humidity_2m",
                "surface_pressure",
            ]
            url = (
                f"https://api.open-meteo.com/v1/gfs?latitude={lat2}&longitude={lon2}"
                f"&hourly={','.join(params)}&timezone=UTC&start_date={date_key}&end_date={date_key}"
            )
            for attempt in range(2):
                try:
                    r = requests.get(url, timeout=15)
                    r.raise_for_status()
                    data = r.json()
                    self._met_cache[key] = {
                        "time": data['hourly']['time'],
                        "temperature_2m": data['hourly'].get('temperature_2m'),
                        "wind_speed_10m": data['hourly'].get('wind_speed_10m'),
                        "wind_direction_10m": data['hourly'].get('wind_direction_10m'),
                        "relative_humidity_2m": data['hourly'].get('relative_humidity_2m'),
                        "surface_pressure": data['hourly'].get('surface_pressure'),
                    }
                    break
                except Exception as e:
                    if attempt == 1:
                        print(f"⚠️ 获取气象失败({lat2},{lon2},{date_key})：{e}")
                    time.sleep(0.5)

        bucket = self._met_cache.get(key)
        if not bucket:
            return {}
        target = timestamp.strftime('%Y-%m-%dT%H:00')
        idx = None
        for i, t in enumerate(bucket['time']):
            if t.startswith(target):
                idx = i
                break
        if idx is None:
            return {}

        out = {
            "temp": None,
            "wind": None,
            "wdir": None,
            "rh": None,
            "pres": None,
        }
        try:
            if bucket['temperature_2m'] is not None:
                out["temp"] = float(bucket['temperature_2m'][idx])
            if bucket['wind_speed_10m'] is not None:
                out["wind"] = float(bucket['wind_speed_10m'][idx])  # m/s
            if bucket['wind_direction_10m'] is not None:
                out["wdir"] = float(bucket['wind_direction_10m'][idx])  # deg (from)
            if bucket['relative_humidity_2m'] is not None:
                out["rh"] = float(bucket['relative_humidity_2m'][idx])  # %
            if bucket['surface_pressure'] is not None:
                p = float(bucket['surface_pressure'][idx])
                out["pres"] = p / 100.0 if p > 2000 else p  # 转 hPa（大于2000基本是 Pa）
        except Exception:
            pass
        return out

    # ---------- 投影：经纬度 -> 像素 ----------
    def _geo_to_pixel_orthographic(self, lat_city, lon_city, img_w, img_h):
        R = float(self.zoom)  # 与 URL zoom 保持一致
        lat0 = math.radians(self.lat)
        lon0 = math.radians(self.lon)
        lat = math.radians(lat_city)
        lon = math.radians(lon_city)

        cos_c = math.sin(lat0) * math.sin(lat) + math.cos(lat0) * math.cos(lat) * math.cos(lon - lon0)
        if cos_c < 0:
            return None  # 背面

        x = R * math.cos(lat) * math.sin(lon - lon0)
        y = R * (math.cos(lat0) * math.sin(lat) - math.sin(lat0) * math.cos(lat) * math.cos(lon - lon0))

        px = int(img_w / 2 + x)
        py = int(img_h / 2 - y)
        return (px, py)

    # ---------- 几何/测量 ----------
    def _measure(self, draw: ImageDraw.ImageDraw, text: str, font):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    @staticmethod
    def _rect_intersect(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(v, hi))

    # ---------- 风向箭头（以城市圆点为锚点，表示“流向”） ----------
    def _draw_wind_arrow(self, draw, x, y, wdir_deg, length=28):
        """
        在城市圆点 (x,y) 处画一支箭，表示风的“流向”（Open-Meteo 返回的是来向 FROM）。
        """
        if wdir_deg is None:
            return
        to_deg = (wdir_deg + 180.0) % 360.0  # 来向->流向
        ang = math.radians(90.0 - to_deg)    # 转为屏幕坐标角
        ex = x + length * math.cos(ang)
        ey = y - length * math.sin(ang)
        draw.line((x, y, ex, ey), fill=(255, 255, 255), width=3)
        # 箭头小翼
        ah = 7
        left = math.radians(90.0 - (to_deg + 20))
        right = math.radians(90.0 - (to_deg - 20))
        lx = ex - ah * math.cos(left)
        ly = ey + ah * math.sin(left)
        rx = ex - ah * math.cos(right)
        ry = ey + ah * math.sin(right)
        draw.line((ex, ey, lx, ly), fill=(255, 255, 255), width=3)
        draw.line((ex, ey, rx, ry), fill=(255, 255, 255), width=3)

    # ---------- 帧捕获 ----------
    def _capture_frame(self, i):
        t = self.start_time + timedelta(hours=i * self.step_hours)
        t_str = t.strftime('%Y-%m-%d-%H')
        timestamp_str = t.strftime('%Y-%m-%d %H:%M UTC')

        url = self._build_url(t)
        self.driver.get(url)
        self._wait_canvas()

        path = os.path.join(self.output_dir, f"frame_{t_str}.png")
        self.driver.save_screenshot(path)

        img = Image.open(path)
        img_cropped = self._crop_view(img)
        self._draw_timestamp(img_cropped, timestamp_str)
        self._draw_overlays(img_cropped, t)

        img_cropped.save(path)
        self.images.append(img_cropped)

    # ---------- 裁剪（基于实际尺寸，水平裁剪以保持中心） ----------
    def _crop_view(self, img):
        W, H = img.size
        crop_width = int(W * 0.60)
        left = (W - crop_width) // 2
        right = left + crop_width
        return img.crop((left, 0, right, H))

    # ---------- 绘制：时间戳 ----------
    def _draw_timestamp(self, img, text):
        draw = ImageDraw.Draw(img)
        margin = 40
        tw, th = self._measure(draw, text, self.font_ts)
        x = img.width - tw - margin
        y = margin
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx or dy:
                    draw.text((x + dx, y + dy), text, font=self.font_ts, fill=(0, 0, 0))
        draw.text((x, y), text, font=self.font_ts, fill=(255, 255, 255))

    # ---------- 绘制：城市叠加（避让 + 多要素 + 箭头） ----------
    def _draw_overlays(self, img, timestamp):
        draw = ImageDraw.Draw(img)
        placed = []  # 已放矩形，做碰撞检测
        r = 10       # 圆点半径
        pad = 6      # 与圆点间距
        rh_label = "湿度" if self.lang == "zh" else "RH"

        for city in self.cities:
            met = self._get_met_at(city['lat'], city['lon'], timestamp) or {}

            # 文本内容（受开关控制）
            parts = [city['name']]
            if self.show_temp:
                temp = met.get('temp')
                temp_s = f"{temp:.1f}°C" if temp is not None else "--°C"
                parts.append(temp_s)
            if self.show_wind:
                wind = met.get('wind')
                wdir = met.get('wdir')
                wind_s = f"{(0.0 if wind is None else wind):.1f} m/s {(0 if wdir is None else int(wdir))}°"
                parts.append(wind_s)
            if self.show_rh:
                rh = met.get('rh')
                rh_s = f"{(0 if rh is None else rh):.0f}%"
                parts.append(f"{rh_label} {rh_s}")
            if self.show_pres and (met.get('pres') is not None):
                pres = met.get('pres')
                parts.append(f"{pres:.0f} hPa")

            label = " | ".join(parts)

            # 投影定位
            xy = self._geo_to_pixel_orthographic(city['lat'], city['lon'], img.width, img.height)
            if xy is None:
                continue
            x, y = xy
            x = self._clamp(x, 12, img.width - 12)
            y = self._clamp(y, 12, img.height - 12)

            # 圆点：外黑描边 + 内白边 + 温度色填充
            color = self._temp_to_color(met.get('temp'))
            draw.ellipse((x - r - 2, y - r - 2, x + r + 2, y + r + 2), outline=(0, 0, 0), width=3)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(255, 255, 255), width=1)

            # 标签避让（右/下/左/上）
            tw, th = self._measure(draw, label, self.font_lbl)
            candidates = [
                (x + r + pad, y - th // 2),        # 右（首选）
                (x - tw // 2, y + r + pad),        # 下
                (x - r - pad - tw, y - th // 2),   # 左
                (x - tw // 2, y - r - pad - th),   # 上
            ]

            placed_rect = None
            tx = ty = None
            for cx, cy in candidates:
                cx = self._clamp(cx, 4, img.width - tw - 4)
                cy = self._clamp(cy, 4, img.height - th - 4)
                rect = (cx - 2, cy - 2, cx + tw + 2, cy + th + 2)
                if all(not self._rect_intersect(rect, pr) for pr in placed):
                    tx, ty = cx, cy
                    placed_rect = rect
                    break

            # 右侧向下级联退避兜底
            if tx is None:
                cx, cy = x + r + pad, y - th // 2
                for k in range(8):
                    cx = self._clamp(cx, 4, img.width - tw - 4)
                    cy = self._clamp(cy + k * (th + 6), 4, img.height - th - 4)
                    rect = (cx - 2, cy - 2, cx + tw + 2, cy + th + 2)
                    if all(not self._rect_intersect(rect, pr) for pr in placed):
                        tx, ty = cx, cy
                        placed_rect = rect
                        break
                if tx is None:
                    tx, ty = self._clamp(x + r + pad, 4, img.width - tw - 4), self._clamp(y - th // 2, 4, img.height - th - 4)
                    placed_rect = (tx, ty, tx + tw, ty + th)

            placed.append(placed_rect)

            # 标签：描边白字
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx or dy:
                        draw.text((tx + dx, ty + dy), label, font=self.font_lbl, fill=(0, 0, 0))
            draw.text((tx, ty), label, font=self.font_lbl, fill=(255, 255, 255))

            # 风向箭头（统一以城市圆点为锚点，表示“流向”）
            if self.show_wind:
                self._draw_wind_arrow(draw, x, y, met.get('wdir'))

    # ---------- 颜色映射 ----------
    @staticmethod
    def _temp_to_color(temp):
        if temp is None:
            return 255, 255, 255
        if temp < 0:
            return 0, 160, 255
        elif temp < 25:
            g = int(255 * (temp / 25))
            return 255 - g, 255, 0
        else:
            r = min(255, int(255 * (temp - 25) / 15))
            return 255, 255 - r, 0

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
        num_frames=8,   # 你原来的跨度
        center_lat=35.0,
        center_lon=105.0,
        zoom=1100,                 # 固定 1100
        window_width=3840,
        window_height=2160,
        output_dir="nullschool_temp_china",
        gif_name="earth_temp_china_ultra_hd.gif",
        mp4_name="earth_temp_china_ultra_hd.mp4",
        delay_per_frame=0.2,
        font_path="/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        font_size=max(22, int(3840 / 160)),  # 或者直接写 24/28/32 等；本版将严格按此值生效
        cities_path="cities.json",  # 外部城市配置（可选）
        lang="zh",                  # "en" 或 "zh"
        show_temp=True,
        show_wind=True,
        show_rh=True,
        show_pres=False
    )

    animator.run_capture()
    animator.export_gif()
    animator.export_mp4()
