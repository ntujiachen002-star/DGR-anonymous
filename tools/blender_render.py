"""Blender headless render script for pipeline figure assets.

Usage:
  blender --background --python tools/blender_render.py

Renders baseline and refined vase meshes with:
- Same camera angle
- Same light gray material
- Studio lighting (3-point)
- White background
- 1024x1024 resolution
- Transparent PNG output
"""
import bpy
import os
import sys
import math

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'figures', 'pipeline_assets')
os.makedirs(OUT, exist_ok=True)

MESHES = {
    'coarse_mesh': os.path.join(ROOT, 'results/mesh_validity_objs/baseline/symmetry/a_symmetric_vase_seed42.obj'),
    'refined_mesh': os.path.join(ROOT, 'results/mesh_validity_objs/handcrafted/symmetry/a_symmetric_vase_seed42.obj'),
}


def clear_scene():
    """Remove all objects."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # Clear orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def setup_render():
    """Configure render settings."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 128
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.film_transparent = True  # Transparent background
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

    # Use denoising for cleaner result
    scene.cycles.use_denoising = True


def setup_camera(target_center=(0, 0, 0), distance=3.0, elevation=25, azimuth=35):
    """Place camera looking at target."""
    elev_rad = math.radians(elevation)
    azim_rad = math.radians(azimuth)

    x = distance * math.cos(elev_rad) * math.sin(azim_rad)
    y = -distance * math.cos(elev_rad) * math.cos(azim_rad)
    z = distance * math.sin(elev_rad)

    cam_data = bpy.data.cameras.new('Camera')
    cam_data.lens = 85  # Portrait lens for less distortion
    cam_obj = bpy.data.objects.new('Camera', cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    cam_obj.location = (x + target_center[0], y + target_center[1], z + target_center[2])

    # Point at target
    direction = mathutils_vector(target_center) - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()

    return cam_obj


def setup_lights():
    """3-point studio lighting."""
    # Key light (main, warm)
    key = bpy.data.lights.new('Key', type='AREA')
    key.energy = 200
    key.size = 3
    key.color = (1.0, 0.98, 0.95)
    key_obj = bpy.data.objects.new('Key', key)
    key_obj.location = (2, -2, 3)
    key_obj.rotation_euler = (math.radians(45), 0, math.radians(45))
    bpy.context.scene.collection.objects.link(key_obj)

    # Fill light (softer, cool)
    fill = bpy.data.lights.new('Fill', type='AREA')
    fill.energy = 80
    fill.size = 4
    fill.color = (0.9, 0.95, 1.0)
    fill_obj = bpy.data.objects.new('Fill', fill)
    fill_obj.location = (-2, -1, 2)
    fill_obj.rotation_euler = (math.radians(40), 0, math.radians(-30))
    bpy.context.scene.collection.objects.link(fill_obj)

    # Rim light (backlight)
    rim = bpy.data.lights.new('Rim', type='AREA')
    rim.energy = 120
    rim.size = 2
    rim.color = (1.0, 1.0, 1.0)
    rim_obj = bpy.data.objects.new('Rim', rim)
    rim_obj.location = (0, 3, 2)
    rim_obj.rotation_euler = (math.radians(-30), 0, math.radians(180))
    bpy.context.scene.collection.objects.link(rim_obj)


def create_material():
    """Create a clean light gray material."""
    mat = bpy.data.materials.new('MeshMaterial')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear defaults
    for node in nodes:
        nodes.remove(node)

    # Principled BSDF
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (0.75, 0.78, 0.82, 1.0)  # Light gray-blue
    bsdf.inputs['Roughness'].default_value = 0.5
    bsdf.inputs['Metallic'].default_value = 0.0
    bsdf.inputs['Specular IOR Level'].default_value = 0.5
    bsdf.location = (0, 0)

    # Output
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    return mat


def import_and_render(obj_path, out_path):
    """Import OBJ, center, normalize, apply material, render."""
    clear_scene()

    # Import OBJ
    bpy.ops.wm.obj_import(filepath=obj_path)

    # Get imported object
    obj = bpy.context.selected_objects[0]

    # Center at origin
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
    obj.location = (0, 0, 0)

    # Normalize scale to fit in unit box
    dims = obj.dimensions
    max_dim = max(dims)
    if max_dim > 0:
        scale = 2.0 / max_dim
        obj.scale = (scale, scale, scale)
        bpy.ops.object.transform_apply(scale=True)

    # Smooth shading
    bpy.ops.object.shade_smooth()

    # Apply material
    mat = create_material()
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    # Setup scene
    setup_lights()

    # Camera
    from mathutils import Vector
    cam_data = bpy.data.cameras.new('Camera')
    cam_data.lens = 85
    cam_obj = bpy.data.objects.new('Camera', cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    dist = 3.5
    elev = math.radians(20)
    azim = math.radians(35)
    cam_obj.location = (
        dist * math.cos(elev) * math.sin(azim),
        -dist * math.cos(elev) * math.cos(azim),
        dist * math.sin(elev)
    )

    # Track to constraint
    track = cam_obj.constraints.new(type='TRACK_TO')
    track.target = obj
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'

    # Render
    setup_render()
    bpy.context.scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    print(f'Rendered: {out_path}')


def main():
    for name, path in MESHES.items():
        if not os.path.exists(path):
            print(f'SKIP {name}: {path} not found')
            continue
        out = os.path.join(OUT, f'{name}_blender.png')
        print(f'\n=== Rendering {name} ===')
        import_and_render(path, out)

    print('\nDone!')


if __name__ == '__main__':
    main()
