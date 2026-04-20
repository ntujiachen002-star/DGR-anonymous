"""Blender render v2: correct orientation (Z-up, vase neck pointing up).
3 views: front, left-side, 3/4 angle (like the reference figure).
Run: blender --background --python tools/blender_render_v2.py
"""
import bpy
import os
import math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "pipeline_assets")

BASELINE_OBJ = os.path.join(ROOT, "results/mesh_validity_objs/baseline/symmetry/a_symmetric_vase_seed42.obj")
REFINED_OBJ  = os.path.join(OUT, "vase_refined_newproto.obj")

# Camera angles: (elevation from XY plane, azimuth around Z, name)
# Vase is Z-up, so:
#   front = camera on +Y axis looking toward origin
#   left  = camera on -X axis looking toward origin
#   3/4   = camera at ~30-40 deg azimuth, slightly elevated (like reference fig)
VIEWS = [
    (15,  0,   "front"),    # Front: camera on +Y, slight elevation
    (15,  90,  "left"),     # Left side: camera on -X
    (25,  35,  "quarter"),  # 3/4 view like reference figure
]


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for col in [bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras]:
        for block in col:
            if block.users == 0:
                col.remove(block)


def setup_render():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 128
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    # White background
    world = bpy.data.worlds.get('World') or bpy.data.worlds.new('World')
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs['Color'].default_value = (1, 1, 1, 1)
        bg.inputs['Strength'].default_value = 0.6


def create_material():
    mat = bpy.data.materials.new('CleanGray')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in nodes:
        nodes.remove(n)
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (0.72, 0.75, 0.80, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.55
    bsdf.inputs['Metallic'].default_value = 0.0
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    return mat


def setup_lights():
    # Key light - upper right
    key = bpy.data.lights.new('Key', 'AREA')
    key.energy = 200
    key.size = 3
    key.color = (1.0, 0.98, 0.95)
    key_obj = bpy.data.objects.new('Key', key)
    key_obj.location = (2, -2.5, 3.5)
    key_obj.rotation_euler = (math.radians(55), math.radians(10), math.radians(35))
    bpy.context.scene.collection.objects.link(key_obj)

    # Fill light - left side
    fill = bpy.data.lights.new('Fill', 'AREA')
    fill.energy = 80
    fill.size = 4
    fill.color = (0.92, 0.95, 1.0)
    fill_obj = bpy.data.objects.new('Fill', fill)
    fill_obj.location = (-2.5, -1, 2.5)
    fill_obj.rotation_euler = (math.radians(50), math.radians(-20), math.radians(-30))
    bpy.context.scene.collection.objects.link(fill_obj)

    # Rim light - behind
    rim = bpy.data.lights.new('Rim', 'AREA')
    rim.energy = 120
    rim.size = 2
    rim.color = (1, 1, 1)
    rim_obj = bpy.data.objects.new('Rim', rim)
    rim_obj.location = (0.5, 3, 2)
    rim_obj.rotation_euler = (math.radians(-20), 0, math.radians(180))
    bpy.context.scene.collection.objects.link(rim_obj)


def render_view(obj_path, prefix, elev_deg, azim_deg, view_name):
    clear_scene()

    # Import OBJ
    bpy.ops.wm.obj_import(filepath=obj_path)
    obj = bpy.context.selected_objects[0]

    # Center + normalize (keep Z-up orientation)
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME')
    obj.location = (0, 0, 0)
    max_dim = max(obj.dimensions)
    if max_dim > 0:
        s = 2.0 / max_dim
        obj.scale = (s, s, s)
        bpy.ops.object.transform_apply(scale=True)

    # Smooth shading
    bpy.ops.object.shade_smooth()

    # Material
    mat = create_material()
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    # Lights
    setup_lights()

    # Camera — spherical coordinates with Z-up
    cam_data = bpy.data.cameras.new('Cam')
    cam_data.lens = 85
    cam_obj = bpy.data.objects.new('Cam', cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    dist = 4.0
    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)

    # Spherical to cartesian (Z-up convention)
    cam_x = dist * math.cos(elev) * math.sin(azim)
    cam_y = -dist * math.cos(elev) * math.cos(azim)  # negative = in front
    cam_z = dist * math.sin(elev)

    cam_obj.location = (cam_x, cam_y, cam_z)

    # Track-to constraint to always look at mesh center
    track = cam_obj.constraints.new(type='TRACK_TO')
    track.target = obj
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'

    # Render
    setup_render()
    out_path = os.path.join(OUT, f'{prefix}_{view_name}_blender.png')
    bpy.context.scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    print(f'  {prefix}_{view_name}_blender.png')


# Render all views for both meshes
for prefix, obj_path in [("baseline", BASELINE_OBJ), ("refined", REFINED_OBJ)]:
    print(f"\n=== {prefix} ===")
    for elev, azim, name in VIEWS:
        render_view(obj_path, prefix, elev, azim, name)

print("\nDone! All renders in:", OUT)
