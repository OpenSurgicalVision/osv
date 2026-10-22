"""M1 synthetic instruments: a grasper and an L-hook rendered with pixel-exact masks.

RESEARCH USE ONLY - NOT FOR CLINICAL USE.

Runs inside Blender's Python (Blender 4.2, or the `bpy==4.2.0` wheel on Python 3.11):

    python scripts/blender/instruments.py --out /outside/the/repo/m1_instruments \\
        --count 200 --seed 1337
    blender --background --python scripts/blender/instruments.py -- --out ... --count 200

The instruments are modelled here from primitives, so the geometry is the project's
own work (no third-party CAD model, no licence to inherit):

* **grasper** - 5 mm insulated shaft, steel collar and clevis, two tapered fenestrated
  jaws opening 0-35 degrees about a hinge;
* **L-hook** - 5 mm insulated shaft tapering to a 1.1 mm steel electrode whose last
  5 mm bend through 90 degrees.

Each sample places one or two instruments (classes drawn independently) in the view
of a 70-degree endoscope, the way they appear in a cholecystectomy: the tip somewhere
in the frame 4-14 cm from the lens, the shaft leaving the frame through its border
towards a trocar. The light sits on the lens axis and falls off with distance, like
the scope's own light; the world is tissue-red, so the metal reflects red.

Per sample, under --out:

* ``<id>_rgba.png`` - the instruments alone, straight alpha (anti-aliased edge);
* ``<id>_class.png`` - one byte per pixel: 0 background, 1 grasper, 2 L-hook. Rendered
  with one sample at the pixel centre, so every pixel has exactly one label and the
  binary instrument mask is ``class > 0``;
* ``<id>_instance.png`` - one byte per pixel: 0 background, 1..n instrument instance;
* ``<id>.json`` - classes, poses, jaw angles, light and seed, enough to re-render.

Compositing over tool-free CholecT50 frames and the photometric augmentations happen in
`osv` (RFC 0001), not here: this script only produces the instruments and their masks.
With --preview it also writes ``<id>_preview.png`` over a procedural tissue-like
background - a placeholder for looking at the render, never training data.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

WIDTH, HEIGHT = 854, 480  # CholecT50 / CholecSeg8k frame size
HFOV_DEG = 70.0
CLASSES = {"grasper": 1, "hook": 2}
MM = 0.001


# --- scene -------------------------------------------------------------------------------


def reset_scene(samples: int) -> bpy.types.Scene:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = WIDTH, HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    scene.render.dither_intensity = 0.0  # dither would put noise into the label render

    camera_data = bpy.data.cameras.new("scope")
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.sensor_width = 36.0
    camera_data.lens = 18.0 / math.tan(math.radians(HFOV_DEG / 2))
    camera_data.clip_start = 0.005
    camera = bpy.data.objects.new("scope", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera  # at the origin, looking down -Z

    world = bpy.data.worlds.new("abdomen")
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.55, 0.22, 0.18, 1.0)
    background.inputs["Strength"].default_value = 0.5
    scene.world = world
    return scene


def add_scope_light(scene: bpy.types.Scene, power_w: float) -> None:
    light_data = bpy.data.lights.new("fibre", type="POINT")
    light_data.energy = power_w
    light_data.shadow_soft_size = 0.002
    light = bpy.data.objects.new("fibre", light_data)
    light.location = (0.0, 0.003, 0.0)  # the fibre ring sits just beside the lens
    scene.collection.objects.link(light)


def material(name: str, color, *, metallic: float, roughness: float) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


# --- geometry: tool frame has the tip at the origin and the shaft along +X ---------------


def _cylinder(name, radius, x0, x1, mat, *, radius_tip=None, vertices=48):
    """A cylinder (or cone) on the X axis between x0 and x1."""
    depth = x1 - x0
    if radius_tip is None:
        bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth)
    else:
        bpy.ops.mesh.primitive_cone_add(
            vertices=vertices, radius1=radius_tip, radius2=radius, depth=depth
        )
    obj = bpy.context.active_object
    obj.name = name
    obj.rotation_euler = (0.0, math.pi / 2, 0.0)  # Z axis -> X axis
    obj.location = ((x0 + x1) / 2, 0.0, 0.0)
    _bake(obj)
    obj.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return obj


def _bake(obj: bpy.types.Object) -> None:
    """Apply location/rotation/scale, so matrix_world is the identity before posing."""
    bpy.context.view_layer.objects.active = obj
    for other in bpy.context.selected_objects:
        other.select_set(False)
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def _jaw(name, length, hinge_x, width0, width1, thickness, opening, mat, upper: bool):
    """A tapered jaw from the hinge towards -X, with a fenestration window."""
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    jaw = bpy.context.active_object
    jaw.name = name
    jaw.scale = (length, width0, thickness)
    jaw.location = (hinge_x - length / 2, 0.0, (thickness / 2) if upper else (-thickness / 2))
    _bake(jaw)
    mesh = jaw.data
    for vertex in mesh.vertices:  # taper towards the tip (smaller x)
        t = (hinge_x - vertex.co.x) / length
        vertex.co.y *= 1.0 + (width1 / width0 - 1.0) * t
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    window = bpy.context.active_object
    window.scale = (length * 0.5, width1 * 0.35, thickness * 4)
    window.location = (hinge_x - length * 0.55, 0.0, 0.0)
    _bake(window)
    boolean = jaw.modifiers.new("fenestration", "BOOLEAN")
    boolean.operation = "DIFFERENCE"
    boolean.object = window
    _bake(jaw)
    bpy.ops.object.modifier_apply(modifier=boolean.name)
    bpy.data.objects.remove(window, do_unlink=True)
    bevel = jaw.modifiers.new("round", "BEVEL")
    bevel.width = thickness * 0.25
    bevel.segments = 3
    bpy.ops.object.modifier_apply(modifier=bevel.name)
    jaw.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    pivot = Matrix.Translation((hinge_x, 0, 0))
    angle = (opening / 2) * (1 if upper else -1)
    jaw.matrix_world = pivot @ Matrix.Rotation(angle, 4, "Y") @ pivot.inverted() @ jaw.matrix_world
    return jaw


def build_grasper(steel, insulation, opening: float) -> list[bpy.types.Object]:
    parts = [
        _cylinder("grasper_shaft", 2.5 * MM, 12 * MM, 330 * MM, insulation),
        _cylinder("grasper_collar", 2.5 * MM, 7 * MM, 12 * MM, steel),
        _cylinder("grasper_clevis", 2.0 * MM, 3 * MM, 7 * MM, steel),
    ]
    for upper in (True, False):
        parts.append(
            _jaw(
                f"grasper_jaw_{'upper' if upper else 'lower'}",
                length=15 * MM,
                hinge_x=4 * MM,
                width0=3.6 * MM,
                width1=2.6 * MM,
                thickness=1.5 * MM,
                opening=opening,
                mat=steel,
                upper=upper,
            )
        )
    return parts


def build_hook(steel, insulation) -> list[bpy.types.Object]:
    parts = [
        _cylinder("hook_shaft", 2.5 * MM, 22 * MM, 330 * MM, insulation),
        _cylinder("hook_taper", 2.5 * MM, 9 * MM, 22 * MM, insulation, radius_tip=0.9 * MM),
        _cylinder("hook_stem", 0.55 * MM, -5 * MM, 9 * MM, steel, vertices=24),
    ]
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.55 * MM, depth=5 * MM)
    bend = bpy.context.active_object
    bend.name = "hook_bend"
    bend.location = (-5 * MM, 0.0, 2.5 * MM)  # the last 5 mm, at 90 degrees to the stem
    _bake(bend)
    bend.data.materials.append(steel)
    bpy.ops.object.shade_smooth()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.55 * MM, location=(-5 * MM, 0, 0))
    corner = bpy.context.active_object
    corner.name = "hook_corner"
    _bake(corner)
    corner.data.materials.append(steel)
    bpy.ops.object.shade_smooth()
    return [*parts, bend, corner]


# --- pose --------------------------------------------------------------------------------


def ray(u: float, v: float) -> Vector:
    """Camera-space direction through pixel (u, v); the camera looks down -Z."""
    half_w = math.tan(math.radians(HFOV_DEG / 2))
    half_h = half_w * HEIGHT / WIDTH
    x = (2 * (u + 0.5) / WIDTH - 1) * half_w
    y = (1 - 2 * (v + 0.5) / HEIGHT) * half_h
    return Vector((x, y, -1.0)).normalized()


def border_pixel(rng: random.Random) -> tuple[float, float]:
    """A point on the image border, sides weighted by length (trocars sit all round)."""
    edge = rng.uniform(0, 2 * (WIDTH + HEIGHT))
    if edge < WIDTH:
        return edge, -1.0
    edge -= WIDTH
    if edge < HEIGHT:
        return WIDTH, edge
    edge -= HEIGHT
    if edge < WIDTH:
        return edge, HEIGHT
    return -1.0, edge - WIDTH


def sample_pose(rng: random.Random) -> dict:
    tip_px = (rng.uniform(0.1, 0.9) * WIDTH, rng.uniform(0.1, 0.9) * HEIGHT)
    tip_depth = rng.uniform(0.04, 0.14)
    entry_px = border_pixel(rng)
    entry_depth = tip_depth + rng.uniform(0.02, 0.10)
    tip = ray(*tip_px) * (tip_depth / -ray(*tip_px).z)
    entry = ray(*entry_px) * (entry_depth / -ray(*entry_px).z)
    return {
        "tip_px": list(tip_px),
        "tip_depth_m": tip_depth,
        "entry_px": list(entry_px),
        "entry_depth_m": entry_depth,
        "roll_rad": rng.uniform(0, 2 * math.pi),
        "tip": list(tip),
        "direction": list((entry - tip).normalized()),
    }


def place(parts: list[bpy.types.Object], pose: dict) -> None:
    direction = Vector(pose["direction"])
    rotation = direction.to_track_quat("X", "Z") @ Quaternion((1, 0, 0), pose["roll_rad"])
    transform = Matrix.Translation(Vector(pose["tip"])) @ rotation.to_matrix().to_4x4()
    for part in parts:
        part.matrix_world = transform @ part.matrix_world


# --- render ------------------------------------------------------------------------------


def _render(scene: bpy.types.Scene, path: Path) -> np.ndarray:
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(str(path), check_existing=False)
    image.colorspace_settings.name = "Non-Color"  # the bytes as written, no sRGB decode
    pixels = np.array(image.pixels[:], dtype=np.float32).reshape(HEIGHT, WIDTH, 4)[::-1]
    bpy.data.images.remove(image)
    return pixels


def render_masks(scene, instances, out: Path, sample_id: str) -> tuple[np.ndarray, np.ndarray]:
    """One centre sample per pixel with flat emission colours: an exact label per pixel."""
    saved = {
        "samples": scene.cycles.samples,
        "denoise": scene.cycles.use_denoising,
        "filter": scene.cycles.filter_width,
        "world": scene.world,
    }
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.cycles.filter_width = 0.01
    scene.world = None
    flats = []
    for number, (class_name, parts) in enumerate(instances, start=1):
        flat = bpy.data.materials.new(f"label_{number}")
        flat.use_nodes = True
        nodes = flat.node_tree.nodes
        nodes.clear()
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = (
            CLASSES[class_name] / 255,
            number / 255,
            0.0,
            1.0,
        )
        output = nodes.new("ShaderNodeOutputMaterial")
        flat.node_tree.links.new(emission.outputs[0], output.inputs[0])
        for part in parts:
            flats.append((part, list(part.data.materials)))
            part.data.materials.clear()
            part.data.materials.append(flat)
    scene.view_settings.view_transform = "Raw"  # label codes pass through unchanged
    pixels = _render(scene, out / f"{sample_id}_labels_raw.png")
    scene.view_settings.view_transform = "Standard"
    (out / f"{sample_id}_labels_raw.png").unlink()
    for part, materials in flats:
        part.data.materials.clear()
        for mat in materials:
            part.data.materials.append(mat)
    scene.cycles.samples = saved["samples"]
    scene.cycles.use_denoising = saved["denoise"]
    scene.cycles.filter_width = saved["filter"]
    scene.world = saved["world"]
    covered = pixels[..., 3] > 0.5
    classes = np.where(covered, np.rint(pixels[..., 0] * 255), 0).astype(np.uint8)
    instance = np.where(covered, np.rint(pixels[..., 1] * 255), 0).astype(np.uint8)
    return classes, instance


def save_gray(array: np.ndarray, path: Path) -> None:
    image = bpy.data.images.new(path.stem, WIDTH, HEIGHT, alpha=False, float_buffer=False)
    rgba = np.zeros((HEIGHT, WIDTH, 4), dtype=np.float32)
    rgba[..., :3] = (array.astype(np.float32) / 255.0)[..., None]
    rgba[..., 3] = 1.0
    image.pixels[:] = rgba[::-1].ravel()
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)


def save_rgb(array: np.ndarray, path: Path) -> None:
    rgba = np.concatenate([array, np.ones((HEIGHT, WIDTH, 1), np.float32)], axis=2)
    image = bpy.data.images.new(path.stem, WIDTH, HEIGHT, alpha=False, float_buffer=False)
    image.pixels[:] = rgba[::-1].ravel()
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)


def placeholder_tissue(rng: np.random.Generator) -> np.ndarray:
    """Smooth red noise inside a dark circular vignette: for previews only."""
    coarse = rng.random((HEIGHT // 24 + 2, WIDTH // 24 + 2))
    ys, xs = (
        np.linspace(0, coarse.shape[0] - 1.001, HEIGHT),
        np.linspace(0, coarse.shape[1] - 1.001, WIDTH),
    )
    y0, x0 = ys.astype(int)[:, None], xs.astype(int)[None, :]
    fy, fx = (ys % 1)[:, None], (xs % 1)[None, :]
    small = (  # bilinear upsampling: smooth blotches, not blocks
        coarse[y0, x0] * (1 - fy) * (1 - fx)
        + coarse[y0 + 1, x0] * fy * (1 - fx)
        + coarse[y0, x0 + 1] * (1 - fy) * fx
        + coarse[y0 + 1, x0 + 1] * fy * fx
    )[..., None]
    base = np.array([0.55, 0.16, 0.12], np.float32) * (0.6 + 0.6 * small)
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    radius = np.hypot((xx - WIDTH / 2) / (WIDTH * 0.62), (yy - HEIGHT / 2) / (HEIGHT * 0.62))
    vignette = np.clip(1.3 - radius, 0, 1)[..., None]
    return np.clip(base * vignette, 0, 1).astype(np.float32)


def render_sample(index: int, seed: int, out: Path, *, samples: int, preview: bool) -> dict:
    rng = random.Random(seed * 100_003 + index)
    scene = reset_scene(samples)
    light_w = rng.uniform(0.8, 2.0)
    add_scope_light(scene, light_w)
    steel = material("steel", (0.90, 0.90, 0.92), metallic=1.0, roughness=rng.uniform(0.25, 0.40))
    insulation = material(
        "insulation", (0.025, 0.025, 0.03), metallic=0.0, roughness=rng.uniform(0.22, 0.40)
    )
    count = 1 if rng.random() < 0.4 else 2
    instances, record = [], []
    for _ in range(count):
        class_name = rng.choice(sorted(CLASSES))
        opening = rng.uniform(0, math.radians(35)) if class_name == "grasper" else 0.0
        parts = (
            build_grasper(steel, insulation, opening)
            if class_name == "grasper"
            else build_hook(steel, insulation)
        )
        pose = sample_pose(rng)
        place(parts, pose)
        for part in parts:
            part.pass_index = CLASSES[class_name]
        instances.append((class_name, parts))
        record.append({"class": class_name, "jaw_opening_rad": opening, **pose})

    sample_id = f"m1s_{seed}_{index:06d}"
    rgba = _render(scene, out / f"{sample_id}_rgba.png")
    classes, instance = render_masks(scene, instances, out, sample_id)
    save_gray(classes, out / f"{sample_id}_class.png")
    save_gray(instance, out / f"{sample_id}_instance.png")
    meta = {
        "id": sample_id,
        "seed": seed,
        "index": index,
        "size": [WIDTH, HEIGHT],
        "hfov_deg": HFOV_DEG,
        "light_w": light_w,
        "samples": samples,
        "blender": bpy.app.version_string,
        "instances": record,
        "pixels": {name: int((classes == value).sum()) for name, value in CLASSES.items()},
    }
    (out / f"{sample_id}.json").write_text(json.dumps(meta, indent=2))
    if preview:
        background = placeholder_tissue(np.random.default_rng(seed + index))
        alpha = rgba[..., 3:4]
        composite = rgba[..., :3] * alpha + background * (1 - alpha)
        save_rgb(composite, out / f"{sample_id}_preview.png")
    return meta


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--start", type=int, default=0, help="first sample index")
    parser.add_argument("--samples", type=int, default=64, help="Cycles samples per pixel")
    parser.add_argument("--preview", action="store_true", help="also write placeholder composites")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for index in range(args.start, args.start + args.count):
        meta = render_sample(index, args.seed, args.out, samples=args.samples, preview=args.preview)
        print(f"[m1-instruments] {meta['id']}: {[i['class'] for i in meta['instances']]}")
    return 0


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    raise SystemExit(main(argv))
