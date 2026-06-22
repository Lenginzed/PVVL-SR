import hashlib
import json
import math
import os
from typing import Dict, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .aircombat_metrics import metrics_for_logging
from .logger import to_jsonable
from .semantic_reward import state_cache_key
from .vlm_prompt import PROMPT_VERSION


RENDERER_VERSION = "situation_renderer_v1"


DEFAULT_RENDERER_CONFIG = {
    "image_size": 512,
    "ego_centered": True,
    "show_legend": True,
    "show_distance": True,
    "show_energy": True,
    "show_speed": True,
    "show_altitude": True,
    "show_delta_energy": True,
    "show_numeric_panel": True,
    "show_attack_sector": True,
    "show_threat_sector": True,
    "view_range_m": 15000.0,
    "prompt_version": PROMPT_VERSION,
}


class SituationRenderer:
    """Render a stable top-down 2D 1v1 air-combat situation image."""

    def __init__(self, renderer_config: Optional[Dict] = None, thresholds: Optional[Dict] = None):
        self.config = dict(DEFAULT_RENDERER_CONFIG)
        if renderer_config:
            self.config.update(renderer_config)
        self.thresholds = thresholds or {}
        self.image_size = int(self.config.get("image_size", 512))
        self.font = ImageFont.load_default()

    def render_sample(
        self,
        state: Dict,
        metrics: Dict,
        labels: Optional[Dict],
        output_dir: str,
        sample_id: Optional[str] = None,
        image_filename: Optional[str] = None,
        metadata_filename: Optional[str] = None,
    ) -> Dict:
        os.makedirs(output_dir, exist_ok=True)
        image = self._draw(state, metrics)
        image_bytes = self._png_bytes(image)
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        image_id = sample_id or image_hash[:16]
        image_path = os.path.abspath(os.path.join(output_dir, image_filename or "{}.png".format(image_id)))
        metadata_path = os.path.abspath(os.path.join(output_dir, metadata_filename or "{}.metadata.json".format(image_id)))
        image.save(image_path)

        metadata = self._metadata(
            state=state,
            metrics=metrics,
            labels=labels,
            image_id=image_id,
            image_hash=image_hash,
            image_path=image_path,
            metadata_path=metadata_path,
        )
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(metadata), f, indent=2, sort_keys=True)
        return metadata

    def _draw(self, state: Dict, metrics: Dict) -> Image.Image:
        size = self.image_size
        center = np.array([size / 2.0, size / 2.0], dtype=np.float64)
        image = Image.new("RGB", (size, size), (248, 249, 250))
        draw = ImageDraw.Draw(image, "RGBA")

        ego_pos = np.asarray(state["ego"]["position"], dtype=np.float64)
        enemy_pos = np.asarray(state["enemy"]["position"], dtype=np.float64)
        origin = ego_pos if self.config.get("ego_centered", True) else 0.5 * (ego_pos + enemy_pos)
        view_range = max(float(self.config.get("view_range_m", 15000.0)), float(metrics["distance"]) * 1.25, 1000.0)
        scale = (size * 0.42) / view_range

        self._draw_grid(draw, center, size)
        self._draw_reference_circles(draw, center, scale, view_range)

        ego_px = self._to_px(ego_pos, origin, center, scale)
        enemy_px = self._to_px(enemy_pos, origin, center, scale)

        if self.config.get("show_attack_sector", True):
            self._draw_sector(
                draw,
                ego_px,
                state["ego"]["heading_vector"],
                float(self.thresholds.get("d_attack_max", 8000.0)) * scale,
                float(self.thresholds.get("theta_launch_deg", 30.0)),
                (28, 113, 216, 44),
                outline=(28, 113, 216, 150),
            )
        if self.config.get("show_threat_sector", True):
            self._draw_sector(
                draw,
                enemy_px,
                state["enemy"]["heading_vector"],
                float(self.thresholds.get("d_missile_max", 14000.0)) * scale,
                float(self.thresholds.get("theta_enemy_launch_deg", 30.0)),
                (220, 53, 69, 62),
                outline=(220, 53, 69, 180),
            )

        self._draw_connection(draw, ego_px, enemy_px, metrics)
        self._draw_aircraft(draw, ego_px, state["ego"], "Ego", (28, 113, 216), scale)
        self._draw_aircraft(draw, enemy_px, state["enemy"], "Enemy", (220, 53, 69), scale)
        self._draw_annotations(draw, metrics, state)
        if self.config.get("show_numeric_panel", True):
            self._draw_numeric_panel(draw, metrics, state)
        return image

    def _draw_grid(self, draw, center, size):
        grid_color = (210, 214, 220, 90)
        step = size // 8
        for x in range(step, size, step):
            draw.line([(x, 0), (x, size)], fill=grid_color, width=1)
            draw.line([(0, x), (size, x)], fill=grid_color, width=1)
        draw.line([(center[0], 0), (center[0], size)], fill=(150, 154, 160, 100), width=1)
        draw.line([(0, center[1]), (size, center[1])], fill=(150, 154, 160, 100), width=1)

    def _draw_reference_circles(self, draw, center, scale, view_range):
        circle_specs = [
            ("tail", float(self.thresholds.get("d_tail", 5000.0)), (34, 139, 34, 70)),
            ("attack", float(self.thresholds.get("d_attack_max", 8000.0)), (28, 113, 216, 70)),
            ("missile", float(self.thresholds.get("d_missile_max", 14000.0)), (220, 53, 69, 60)),
        ]
        for name, radius_m, color in circle_specs:
            radius = radius_m * scale
            if radius <= 3 or radius > self.image_size:
                continue
            bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
            draw.ellipse(bbox, outline=color, width=2)
            draw.text((center[0] + radius + 4, center[1] - 8), name, fill=color[:3], font=self.font)

    def _draw_sector(self, draw, origin_px, heading_vector, radius_px, half_angle_deg, fill, outline=None):
        if radius_px <= 1:
            return
        angle = self._image_angle(heading_vector)
        half = math.radians(half_angle_deg)
        points = [tuple(origin_px)]
        for theta in np.linspace(angle - half, angle + half, 28):
            points.append((origin_px[0] + radius_px * math.cos(theta), origin_px[1] + radius_px * math.sin(theta)))
        draw.polygon(points, fill=fill)
        if outline:
            left = origin_px + radius_px * np.array([math.cos(angle - half), math.sin(angle - half)])
            right = origin_px + radius_px * np.array([math.cos(angle + half), math.sin(angle + half)])
            draw.line([tuple(origin_px), tuple(left)], fill=outline, width=2)
            draw.line([tuple(origin_px), tuple(right)], fill=outline, width=2)
            draw.arc(
                [
                    origin_px[0] - radius_px,
                    origin_px[1] - radius_px,
                    origin_px[0] + radius_px,
                    origin_px[1] + radius_px,
                ],
                start=math.degrees(angle - half),
                end=math.degrees(angle + half),
                fill=outline,
                width=2,
            )

    def _draw_connection(self, draw, ego_px, enemy_px, metrics):
        draw.line([tuple(ego_px), tuple(enemy_px)], fill=(60, 64, 67, 150), width=2)
        if self.config.get("show_distance", True):
            mid = 0.5 * (ego_px + enemy_px)
            draw.text(tuple(mid + np.array([8, -12])), "{:.1f} km".format(metrics["distance"] / 1000.0), fill=(35, 35, 35), font=self.font)

    def _draw_aircraft(self, draw, point, aircraft, label, color, scale):
        x, y = float(point[0]), float(point[1])
        radius = 8
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(*color, 255), outline=(20, 20, 20, 255), width=1)
        self._draw_arrow(draw, point, aircraft["heading_vector"], 52, (*color, 255), width=4)
        self._draw_dashed_arrow(draw, point, aircraft["velocity"], 42, (*color, 190), width=2)
        draw.text((x + 10, y + 8), label, fill=color, font=self.font)

    def _draw_arrow(self, draw, origin_px, vector, length, fill, width=3):
        direction = self._dir2d(vector)
        end = origin_px + direction * length
        draw.line([tuple(origin_px), tuple(end)], fill=fill, width=width)
        self._draw_arrow_head(draw, end, direction, fill)

    def _draw_dashed_arrow(self, draw, origin_px, vector, length, fill, width=2):
        direction = self._dir2d(vector)
        if np.linalg.norm(direction) < 1e-8:
            return
        segment = 8
        gap = 5
        pos = 0
        while pos < length:
            start = origin_px + direction * pos
            end = origin_px + direction * min(pos + segment, length)
            draw.line([tuple(start), tuple(end)], fill=fill, width=width)
            pos += segment + gap
        self._draw_arrow_head(draw, origin_px + direction * length, direction, fill, size=7)

    def _draw_arrow_head(self, draw, end, direction, fill, size=9):
        angle = math.atan2(direction[1], direction[0])
        left = angle + math.pi * 0.82
        right = angle - math.pi * 0.82
        p1 = end
        p2 = end + size * np.array([math.cos(left), math.sin(left)])
        p3 = end + size * np.array([math.cos(right), math.sin(right)])
        draw.polygon([tuple(p1), tuple(p2), tuple(p3)], fill=fill)

    def _draw_annotations(self, draw, metrics, state):
        y = 12
        draw.rectangle([8, 8, 268, 108], fill=(255, 255, 255, 210), outline=(200, 200, 200, 180))
        legend = [
            "Blue = Ego, Red = Enemy",
            "Solid arrow = heading",
            "Dashed arrow = velocity",
        ]
        if self.config.get("show_energy", True):
            legend.append("Delta E: {:.0f} m2/s2".format(metrics["energy_difference"]))
            legend.append("Delta h: {:.0f} m".format(state["ego"]["altitude"] - state["enemy"]["altitude"]))
        for line in legend:
            draw.text((14, y), line, fill=(30, 30, 30), font=self.font)
            y += 18

    def _draw_numeric_panel(self, draw, metrics, state):
        lines = []
        if self.config.get("show_speed", True):
            lines.extend([
                "Ego V: {:.0f} m/s".format(state["ego"]["speed"]),
                "Enemy V: {:.0f} m/s".format(state["enemy"]["speed"]),
            ])
        if self.config.get("show_altitude", True):
            lines.extend([
                "Ego H: {:.0f} m".format(state["ego"]["altitude"]),
                "Enemy H: {:.0f} m".format(state["enemy"]["altitude"]),
            ])
        if self.config.get("show_delta_energy", True):
            lines.append("DeltaE: {:.0f}".format(metrics["energy_difference"]))
        if not lines:
            return
        width = 156
        height = 14 + 18 * len(lines)
        x0 = max(8, self.image_size - width - 8)
        y0 = 8
        draw.rectangle([x0, y0, x0 + width, y0 + height], fill=(255, 255, 255, 218), outline=(175, 180, 185, 190))
        y = y0 + 8
        for line in lines:
            draw.text((x0 + 8, y), line, fill=(28, 32, 36), font=self.font)
            y += 18

    def _metadata(self, state, metrics, labels, image_id, image_hash, image_path, metadata_path):
        state_hash = state_cache_key(state)
        return {
            "image_id": image_id,
            "image_hash": image_hash,
            "state_hash": state_hash,
            "cache_key": state_hash,
            "image_path": image_path,
            "metadata_path": metadata_path,
            "renderer_version": RENDERER_VERSION,
            "prompt_version": self.config.get("prompt_version", PROMPT_VERSION),
            "renderer_config": self.config,
            "coordinate_system": "top_down_local_NEU_ego_centered_pixels_x_east_y_north",
            "agent_id": state.get("agent_id"),
            "enemy_id": state.get("enemy_id"),
            "metrics": metrics_for_logging(metrics),
            "labels": labels or {},
        }

    def _to_px(self, position, origin, center, scale):
        relative = np.asarray(position, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
        # Local position is NEU: [north, east, up]. Image x=east, y=-north.
        return center + np.array([relative[1], -relative[0]]) * scale

    def _dir2d(self, vector):
        vector = np.asarray(vector, dtype=np.float64)
        direction = np.array([vector[1], -vector[0]], dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm <= 1e-8:
            return np.zeros(2, dtype=np.float64)
        return direction / norm

    def _image_angle(self, heading_vector):
        direction = self._dir2d(heading_vector)
        if np.linalg.norm(direction) <= 1e-8:
            return 0.0
        return math.atan2(direction[1], direction[0])

    def _png_bytes(self, image):
        from io import BytesIO

        buf = BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
