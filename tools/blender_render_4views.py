"""Blender headless render: baseline + refined vase, 4 views each.
Run: blender --background --python tools/blender_render_4views.py
"""
import bpy
import os
import math
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "pipeline_assets")

BASELINE_OBJ = os.path.join(ROOT, "results/mesh_validity_objs/baseline/symmetry/a_symmetric_vase_seed42.obj")
REFINED_OBJ  = os.path.join(OUT, "vase_refined_newproto.obj")

ANGLES = [(20, 0, "front"), (20, 90, "right"), (20, 180, "back"), (20, 270, "left")]


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.lights:
        if block.users == 0:
            bpy.data.lights.remove(block)
    for block in bpy.data.cameras:
        if block.users == 0:
            bpy.data.cameras.remove(block)


def setup_render():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 64
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
        bg.inputs['Strength'].default_value = 0.5


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
    bsdf.location = (0, 0)

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    return mat


def setup_lights():
    # Key
    key_data = bpy.data.lights.new('Key', 'AREA')
    key_data.energy = 150
    key_data.size = 3
    key_data.color = (1.0, 0.98, 0.95)
    key_obj = bpy.data.objects.new('Key', key_data)
    key_obj.location = (2.5, -2, 3)
    key_obj.rotation_euler = (math.radians(50), 0, math.radians(40))
    bpy.context.scene.collection.objects.link(key_obj)

    # Fill
    fill_data = bpy.data.lights.new('Fill', 'AREA')
    fill_data.energy = 60
    fill_data.size = 4
    fill_data.color = (0.92, 0.95, 1.0)
    fill_obj = bpy.data.objects.new('Fill', fill_data)
    fill_obj.location = (-2.5, -1, 2)
    fill_obj.rotation_euler = (math.radians(45), 0, math.radians(-35))
    bpy.context.scene.collection.objects.link(fill_obj)

    # Rim
    rim_data = bpy.data.lights.new('Rim', 'AREA')
    rim_data.energy = 100
    rim_data.size = 2
    rim_data.color = (1, 1, 1)
    rim_obj = bpy.data.objects.new('Rim', rim_data)
    rim_obj.location = (0, 3, 1.5)
    rim_obj.rotation_euler = (math.radians(-25), 0, math.radians(180))
    bpy.context.scene.collection.objects.link(rim_obj)


def import_and_render(obj_path, prefix):
    for elev, azim, view_name in ANGLES:
        clear_scene()

        # Import
        bpy.ops.wm.obj_import(filepath=obj_path)
        obj = bpy.context.selected_objects[0]

        # Center + normalize
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME')
        obj.location = (0, 0, 0)
        dims = obj.dimensions
        max_dim = max(dims)
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

        # Camera
        cam_data = bpy.data.cameras.new('Cam')
        cam_data.lens = 85
        cam_obj = bpy.data.objects.new('Cam', cam_data)
        bpy.context.scene.collection.objects.link(cam_obj)
        bpy.context.scene.camera = cam_obj

        dist = 3.8
        er, ar = math.radians(elev), math.radians(azim)
        cam_obj.location = (
            dist * math.cos(er) * math.sin(ar),
            -dist * math.cos(er) * math.cos(ar),
            dist * math.sin(er)
        )
        track = cam_obj.constraints.new(type='TRACK_TO')
        track.target = obj
        track.track_axis = 'TRACK_NEGATIVE_Z'
        track.up_axis = 'UP_Y'

        # Render
        setup_render()
        out_path = os.path.join(OUT, f'{prefix}_{view_name}_blender.png')
        bpy.context.scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        print(f'  Rendered: {prefix}_{view_name}_blender.png')


print("=== Rendering baseline (4 views) ===")
import_and_render(BASELINE_OBJ, "baseline")

print("\n=== Rendering refined (4 views) ===")
import_and_render(REFINED_OBJ, "refined")

print("\nDone!")
