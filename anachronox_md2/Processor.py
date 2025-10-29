import bpy
from bpy.props import IntProperty, BoolProperty

from dataclasses import dataclass, fields
from typing import List
import math
from mathutils import Vector

from .utils import *



def flatten_extend(matrix):
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list

def insert_keyframe(fcurves, frame, values):
    for fcu, val in zip(fcurves, values):
        fcu.keyframe_points.insert(frame, val, options={'FAST'})


class ImportAnimationFrames(bpy.types.Operator):
    bl_idname = "wm.import_animation_frames"
    bl_label = "Import Animation Frames"

    def execute(self, context):
        self.finished = False

        frames = ModelVars.my_object.frames
        frame_count = len(frames)
        if frame_count == 0:
            return {'CANCELLED'}

        obj = ModelVars.obj
        mesh = obj.data

        # Ensure mesh has its own animation container (we will put Actions here)
        mesh.animation_data_create()
        # Make sure object.animation_data exists too but keep object.action None so NLA on mesh is authoritative
        obj.animation_data_create()
        obj.animation_data.action = None

        ModelVars.current_anim_name = None

        current_action = None
        current_chunk_start_global = None
        current_chunk_length = 0
        created_tracks = []

        print("Looping through frames...")

        orig_mode = None
        if bpy.context.object:
            orig_mode = bpy.context.object.mode

        def _ensure_fcurve(action, data_path, index):
            fcu = action.fcurves.find(data_path, index=index)
            if fcu is None:
                fcu = action.fcurves.new(data_path=data_path, index=index)
            return fcu

        def _insert_vertex_into_action(action, vert_index, co, frame):
            data_path = f"vertices[{vert_index}].co"    # note: on mesh action, data_path is relative to the Mesh ID
            for i in range(3):
                fcu = _ensure_fcurve(action, data_path, i)
                fcu.keyframe_points.insert(frame, co[i], options={'FAST'})

        for frame_index, frame in enumerate(frames):
            # if (frame_index % 10 == 0):
                # showProgress(frame_index, frame_count)

            global_frame = frame_index * 2
            anim_name = frame.name[: findnth(frame.name, '_', 2)]

            if anim_name != ModelVars.current_anim_name:
                print(f"Processing animation: {anim_name} - Frame #: {frame_index}")

                if not anim_name == '':
                    # finalize previous action (normalize fcurves and push to mesh NLA)
                    if current_action is not None:
                        for fcu in current_action.fcurves:
                            if not fcu.keyframe_points:
                                continue
                            min_x = min(kp.co.x for kp in fcu.keyframe_points)
                            if min_x != 0.0:
                                for kp in fcu.keyframe_points:
                                    kp.co.x -= min_x
                                fcu.update()

                        # clear mesh active action so mesh NLA is authoritative
                        mesh.animation_data.action = None

                        # create NLA track+strip on mesh.animation_data
                        ad = mesh.animation_data
                        track = ad.nla_tracks.new()
                        track.name = f"{ModelVars.object_name}_{ModelVars.current_anim_name}_track"
                        strip = track.strips.new(name=current_action.name, start=int(0), action=current_action)
                        strip.action_frame_start = 0
                        strip.action_frame_end = int(current_chunk_length)
                        strip.frame_end = int(current_chunk_length)
                        created_tracks.append(track)

                        # mute other mesh tracks, unmute this one for clarity
                        for t in ad.nla_tracks:
                            t.mute = True
                        track.mute = False

                    # create new Action on the mesh datablock and set it active there
                    act_name = f"{ModelVars.object_name}_{anim_name}"
                    current_action = bpy.data.actions.new(name=act_name)
                    mesh.animation_data.action = current_action   # assign action to mesh (mesh ID)

                else:
                    current_action = None
                    print("SKIPPING BLANK ANIMATION NAME...")

                # Do this whether it's a blank animation or not
                current_chunk_start_global = global_frame
                current_chunk_length = 0
                ModelVars.current_anim_name = anim_name


            local_frame = int(global_frame - current_chunk_start_global)

            if current_action is not None:
                # fast write vertex coords
                mesh.vertices.foreach_set('co', flatten_extend(ModelVars.all_verts[frame_index]))

                # ensure OBJECT mode for safe operations
                if bpy.context.object and bpy.context.object.mode != 'OBJECT':
                    bpy.ops.object.mode_set(mode='OBJECT')

                # insert keyframes into the mesh-level action fcurves
                for idx, v in enumerate(mesh.vertices):
                    v.co = ModelVars.all_verts[frame_index][idx]
                    _insert_vertex_into_action(current_action, idx, v.co, local_frame)

            current_chunk_length = max(current_chunk_length, local_frame)

        # finalize last chunk
        if current_action is not None:
            for fcu in current_action.fcurves:
                if not fcu.keyframe_points:
                    continue
                min_x = min(kp.co.x for kp in fcu.keyframe_points)
                if min_x != 0.0:
                    for kp in fcu.keyframe_points:
                        kp.co.x -= min_x
                    fcu.update()

            mesh.animation_data.action = None

            ad = mesh.animation_data
            track = ad.nla_tracks.new()
            track.name = f"{ModelVars.object_name}_{ModelVars.current_anim_name}_track"
            strip = track.strips.new(name=current_action.name, start=int(0), action=current_action)
            strip.action_frame_start = 0
            strip.action_frame_end = int(current_chunk_length)
            strip.frame_end = int(current_chunk_length)
            created_tracks.append(track)

            for t in ad.nla_tracks:
                t.mute = True
            track.mute = False


        # unmute only the first track, mute the rest
        for i, t in enumerate(created_tracks):
            t.mute = (i != 0)

        # Find an area/region of type 'NLA' to provide context for the operator
        nla_area = None
        for area in bpy.context.window.screen.areas:
            if area.type == 'NLA':
                nla_area = area
                break

        # restore mode
        if orig_mode and bpy.context.object and bpy.context.object.mode != orig_mode:
            bpy.ops.object.mode_set(mode=orig_mode)

        # set scene range to longest action length
        max_len = 0
        for t in created_tracks:
            for s in t.strips:
                max_len = max(max_len, int(s.frame_end))
        if max_len > 0:
            bpy.context.scene.frame_start = 0
            bpy.context.scene.frame_end = max_len

        # showProgress(frame_count, frame_count, "Import complete.")
        return {'FINISHED'}





class ImportMaterials(bpy.types.Operator):
    bl_idname = "wm.import_materials"
    bl_label = "Import Materials"

    def execute(self, context):
        self.finished = False

        # New method to assign materials per skin/texture (we only have the single triangle reference for tagged surfaces, and they are not 1-to-1 with skins)
        texpath=0
        print("***MATERIALS***")
        for skin_index, skin in enumerate(ModelVars.my_object.skin_names):

            material_name = ("M_" + ModelVars.my_object.skin_names[skin_index].rstrip("\x00"))
            print(f"Blender Material name: {material_name}")

            mat = bpy.data.materials.new(name=material_name)
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes["Principled BSDF"]

            if (bpy.app.version < (4,0,0)):
                bsdf.inputs['Specular'].default_value = 0
            else:
                bsdf.inputs['Specular IOR Level'].default_value = 0

            texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')

            # Give an error and assign a purple color if all textures are missing
            if(ModelVars.my_object.texture_paths == []):
                # Give and error and assign a purple color if one texture is missing
                print(f"Cannot find textures for {ImportOptions.md2_path}!")
                print(f"Check {ModelVars.model_path} for .mda material texture file.")
                bsdf.inputs['Base Color'].default_value = (1,0,1,1)

            if(texpath < len(ModelVars.my_object.texture_paths)):
                # Give and error and assign a purple color if one texture is missing
                if(ModelVars.my_object.texture_paths[skin_index] == ''):
                    print(f"Material Texture: MISSING!")
                    bsdf.inputs['Base Color'].default_value = (1,0,1,1)

                else:
                    print(f"Material Texture: {ModelVars.my_object.texture_paths[skin_index]}")
                if ModelVars.my_object.texture_paths[skin_index] != '':
                    texImage.image = bpy.data.images.load(ModelVars.my_object.texture_paths[skin_index])
                    mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
                    # again copy and paste
                else:
                    # print(f"Cannot find texture {ModelVars.my_object.texture_paths[triangle_index]}!")
                    print(f"Check {ModelVars.model_path} for .mda material texture file.")
                    bsdf.inputs['Base Color'].default_value = (1,0,1,1)

            # Fall back to purple if we missed something or a triangle doesn't have a material assigned.
            bsdf.inputs['Base Color'].default_value = (1,0,1,1)

            texpath += 1
            mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
            ModelVars.obj.data.materials.append(mat)



        # If there are multiple textures, we need to reassign the triangles
        if len(ModelVars.my_object.skin_names) > 1:
            bpy.context.tool_settings.mesh_select_mode = [False, False, True]

            for material_index, material in enumerate(ModelVars.obj.data.materials):
                bpy.context.object.active_material_index = material_index

                # print(f"Current material index: {bpy.context.object.active_material_index}")
                bpy.ops.object.mode_set(mode = 'EDIT')
                bpy.ops.mesh.select_all(action = 'DESELECT')

                skin_triangle_list = list()
                # triangle is key, skin index is the value
                for tri in ModelVars.my_object.triangle_skin_dict:
                    if ModelVars.my_object.triangle_skin_dict[tri] == material_index:
                        # print(f"Appending triangle {tri} to list for skin {material_index}")
                        skin_triangle_list.append(tri)

                bpy.ops.object.mode_set(mode = 'OBJECT')
                for face_idx, face in enumerate(ModelVars.mesh.polygons):
                    # mesh.polygons[face_idx].select = True
                    if face_idx in (skin_triangle_list):
                        face.material_index = bpy.context.object.active_material_index

                # bpy.ops.object.mode_set(mode = 'EDIT')
                # bpy.ops.object.material_slot_assign()


        # Apply new scale set on import screen
        # bpy.ops.view3d.snap_cursor_to_center({'area':view3d})
        # bpy.ops.transform.translate(value=(0, 0, 1), orient_type='GLOBAL')

        # put cursor at origin
        # bpy.context.scene.cursor.location = Vector((0.0, 0.0, 0.0))
        # bpy.context.scene.cursor.rotation_euler = Vector((0.0, 0.0, 0.0))

        print("Seting to object mode...")
        bpy.ops.object.mode_set(mode = 'OBJECT')
        print("Setting origin to geometry...")
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
        # print("Setting origin to cursor...")
        # bpy.ops.object.origin_set(type='ORIGIN_CURSOR')

        # REMOVED - This doesn't work on all frames ********************************
        # Any rotation needs to be applied to all frames
        # for frame_index, frame in enumerate(ModelVars.my_object.frames):
        #     bpy.context.scene.frame_set(frame_index)
        #     bpy.context.active_object.rotation_euler[0] = math.radians(ImportOptions.x_rotate) # rotate on import axis=(1=X 2=Y, 3=Z) degrees=(amount)
        #     bpy.context.active_object.rotation_euler[1] = math.radians(ImportOptions.y_rotate) # rotate on import axis=(1=X 2=Y, 3=Z) degrees=(amount)
        #     bpy.context.active_object.rotation_euler[2] = math.radians(ImportOptions.z_rotate) # rotate on import axis=(1=X 2=Y, 3=Z) degrees=(amount)

        # Reset
        bpy.context.scene.frame_set(0)
        print("Object rotated per selected parameters...")

        # Apply Transforms if option selected on import screen
        if(ImportOptions.apply_transforms):
            print("Applying transforms...")
            context = bpy.context
            ob = context.object
            mb = ob.matrix_basis
            if hasattr(ob.data, "transform"):
                ob.data.transform(mb)
            for c in ob.children:
                c.matrix_local = mb @ c.matrix_local

            ob.matrix_basis.identity()     

        # Apply flip if option is selected on import screen
        if(ImportOptions.recalc_normals):
            print("Recalculating normals...")
            # go edit mode
            bpy.ops.object.mode_set(mode='EDIT')
            # select al faces
            bpy.ops.mesh.select_all(action='SELECT')#Change to select object just made
            # bpy.ops.mesh.flip_normals() # just flip normals
            bpy.ops.mesh.normals_make_consistent(inside=False) # recalculate outside
            # bpy.ops.mesh.normals_make_consistent(inside=True) # recalculate inside
            # go object mode again
            bpy.ops.object.editmode_toggle()

        print("YAY NO ERRORS!!")
        return {'FINISHED'}







