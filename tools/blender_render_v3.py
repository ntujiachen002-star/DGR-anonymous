import os
"""Blender render v3: fix vase orientation + 3 views (front, left, 3/4).
Run: blender --background --python tools/blender_render_v3.py
"""
import bpy, os, math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "pipeline_assets")

BASELINE = os.path.join(ROOT, "results/mesh_validity_objs/baseline/symmetry/a_symmetric_vase_seed42.obj")
REFINED  = os.path.join(OUT, "vase_refined_newproto.obj")

# Views: camera position in spherical coords relative to upright vase
# After import, vase Z-up in OBJ becomes -Y-forward in Blender (Blender Z-up convention)
# So we need: front = camera looks along +Y, left = camera looks along +X
VIEWS = [
    # (cam_x, cam_y, cam_z, name)
    (0, -5, 1.5,       "front"),       # Front: straight on
    (-3.5, -3.5, 2.0,  "front_right"), # Front-right 3/4
    (3.5, 3.5, 2.0,    "back_left"),   # Back-left 3/4
]


def clear_all():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for c in [bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras]:
        for b in c:
            if b.users == 0: c.remove(b)


def setup():
    s = bpy.context.scene
    s.render.engine = 'CYCLES'
    s.cycles.device = 'CPU'
    s.cycles.samples = 128
    s.render.resolution_x = 1024
    s.render.resolution_y = 1024
    s.render.film_transparent = False
    s.render.image_settings.file_format = 'PNG'
    w = bpy.data.worlds.get('World') or bpy.data.worlds.new('World')
    s.world = w
    w.use_nodes = True
    w.node_tree.nodes.get('Background').inputs['Color'].default_value = (1,1,1,1)
    w.node_tree.nodes.get('Background').inputs['Strength'].default_value = 0.5


def make_mat():
    m = bpy.data.materials.new('Gray')
    m.use_nodes = True
    n = m.node_tree.nodes
    for x in n: n.remove(x)
    b = n.new('ShaderNodeBsdfPrincipled')
    b.inputs['Base Color'].default_value = (0.53, 0.81, 0.92, 1)  # sky blue like user study
    b.inputs['Roughness'].default_value = 0.45
    o = n.new('ShaderNodeOutputMaterial')
    o.location = (300, 0)
    m.node_tree.links.new(b.outputs['BSDF'], o.inputs['Surface'])
    return m


def add_lights():
    for name, typ, energy, sz, color, loc, rot in [
        ('Key', 'AREA', 200, 3, (1,.98,.95), (2,-3,4), (55,10,35)),
        ('Fill', 'AREA', 80, 4, (.92,.95,1), (-3,-1,3), (50,-20,-30)),
        ('Rim', 'AREA', 120, 2, (1,1,1), (0,3,2.5), (-20,0,180)),
    ]:
        d = bpy.data.lights.new(name, typ)
        d.energy = energy
        d.size = sz
        d.color = color
        o = bpy.data.objects.new(name, d)
        o.location = loc
        o.rotation_euler = tuple(math.radians(r) for r in rot)
        bpy.context.scene.collection.objects.link(o)


def do_render(obj_path, prefix):
    for cx, cy, cz, view in VIEWS:
        clear_all()

        # Import with correct forward/up axes
        # OBJ has Z-up, Y-forward; Blender default import should handle this
        bpy.ops.wm.obj_import(
            filepath=obj_path,
            forward_axis='NEGATIVE_Y',
            up_axis='Z'
        )
        obj = bpy.context.selected_objects[0]

        # Center
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME')
        obj.location = (0, 0, 0)

        # Scale to fit
        mx = max(obj.dimensions)
        if mx > 0:
            s = 2.0 / mx
            obj.scale = (s, s, s)
            bpy.ops.object.transform_apply(scale=True)

        bpy.ops.object.shade_smooth()

        mat = make_mat()
        obj.data.materials.clear()
        obj.data.materials.append(mat)

        add_lights()

        # Camera
        cd = bpy.data.cameras.new('C')
        cd.lens = 50
        co = bpy.data.objects.new('C', cd)
        co.location = (cx, cy, cz)
        bpy.context.scene.collection.objects.link(co)
        bpy.context.scene.camera = co

        t = co.constraints.new('TRACK_TO')
        t.target = obj
        t.track_axis = 'TRACK_NEGATIVE_Z'
        t.up_axis = 'UP_Y'

        setup()
        out = os.path.join(OUT, f"{prefix}_{view}_blender.png")
        bpy.context.scene.render.filepath = out
        bpy.ops.render.render(write_still=True)
        print(f"  {prefix}_{view}")


print("=== Baseline ===")
do_render(BASELINE, "baseline")
print("\n=== Refined ===")
do_render(REFINED, "refined")
print("\nDone!")
