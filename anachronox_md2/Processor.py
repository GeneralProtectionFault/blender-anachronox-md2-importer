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



class IMPORT_OT_animation_frames_modal(bpy.types.Operator):
    bl_idname = "wm.import_animation_frames_modal"
    bl_label = "Import Animation Frames (Modal)"
    bl_options = {'REGISTER', 'UNDO'}  # UNDO kept optional; modal + long ops normally don't support automatic undo fully

    # internal state (not properties)
    _timer = None
    _frames = None
    _frame_count = 0
    _frame_index = 0
    _obj = None
    _mesh = None
    _orig_mode = None

    # current action/chunk state
    _current_action = None
    _current_chunk_start_global = None
    _current_chunk_length = 0
    _created_tracks = None

    def _ensure_fcurve(self, action, data_path, index):
        if (bpy.app.version < (4,4,0)):
            fcu = action.fcurves.find(data_path, index=index)
            if fcu is None:
                fcu = action.fcurves.new(data_path=data_path, index=index)
        else:                                   # Blender 5+
            slot = action.slots[0]
            strip = action.layers[0].strips[0]
            channelbag = strip.channelbag(slot)
            fcu = channelbag.fcurves.find(data_path, index=index)
            if fcu is None:
                fcu = channelbag.fcurves.new(data_path=data_path, index=index)

        return fcu

    def _insert_vertex_into_action(self, action, vert_index, co, frame):
        data_path = f"vertices[{vert_index}].co"
        for i in range(3):
            fcu = self._ensure_fcurve(action, data_path, i)
            fcu.keyframe_points.insert(frame, co[i], options={'FAST'})

    def _finalize_current_action(self):
        mesh = self._mesh
        if self._current_action is None:
            return

        if (bpy.app.version < (4,4,0)):
            fcurves = self._current_action.fcurves
        else:
            slot = self._current_action.slots[0]
            strip = self._current_action.layers[0].strips[0]
            channelbag = strip.channelbag(slot)
            fcurves = channelbag.fcurves

        for fcu in fcurves:
            if not fcu.keyframe_points:
                continue
            min_x = min(kp.co.x for kp in fcu.keyframe_points)
            if min_x != 0.0:
                for kp in fcu.keyframe_points:
                    kp.co.x -= min_x
                fcu.update()

        # clear mesh active action so mesh NLA is authoritative
        mesh.animation_data.action = None
        ad = mesh.animation_data
        track = ad.nla_tracks.new()
        track.name = f"{ModelVars.object_name}_{ModelVars.current_anim_name}_track"
        strip = track.strips.new(name=self._current_action.name, start=int(0), action=self._current_action)
        strip.action_frame_start = 0
        strip.action_frame_end = int(self._current_chunk_length)
        strip.frame_end = int(self._current_chunk_length)
        self._created_tracks.append(track)
        for t in ad.nla_tracks:
            t.mute = True
        track.mute = False

    def execute(self, context):
        frames = ModelVars.my_object.frames
        frame_count = len(frames)
        if frame_count == 0:
            self.report({'WARNING'}, "No frames to import")
            return {'CANCELLED'}

        self._frames = frames
        self._frame_count = frame_count
        self._frame_index = 0
        self._obj = ModelVars.obj
        self._mesh = self._obj.data
        self._orig_mode = None
        if bpy.context.object:
            self._orig_mode = bpy.context.object.mode

        # Prepare animation containers
        self._mesh.animation_data_create()
        self._obj.animation_data_create()
        self._obj.animation_data.action = None
        ModelVars.current_anim_name = None

        self._current_action = None
        self._current_chunk_start_global = None
        self._current_chunk_length = 0
        self._created_tracks = []

        # Start modal timer
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)

        # Begin progress
        context.window_manager.progress = 0.0
        context.window_manager.progress_message = "Animations:"

        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        wm = context.window_manager

        if event.type == 'TIMER':
            # process a small batch per tick to keep UI responsive
            batch_size = 5  # adjust for performance; small number keeps UI interactive
            processed = 0
            frames = self._frames

            while self._frame_index < self._frame_count and processed < batch_size:
                frame_index = self._frame_index
                frame = frames[frame_index]

                global_frame = frame_index * 2
                anim_name = frame.name[: findnth(frame.name, '_', 2)]

                if anim_name != ModelVars.current_anim_name:
                    print(f"Processing animation: {anim_name} - Frame #: {frame_index}")

                    if not anim_name == '':
                        # finalize previous action
                        if self._current_action is not None:
                            self._finalize_current_action()

                        # create new Action on the mesh datablock and set it active there
                        act_name = f"{ModelVars.object_name}_{anim_name}"
                        self._current_action = bpy.data.actions.new(name=act_name)
                        self._mesh.animation_data.action = self._current_action

                        if (bpy.app.version >= (4,4,0)):
                            self._current_action.slots.new(id_type='MESH', name=act_name)
                            slot = self._current_action.slots[0]
                            layer = self._current_action.layers.new("Base")
                            strip = layer.strips.new(type='KEYFRAME')
                            channelbag = strip.channelbag(slot, ensure=True)

                    else:
                        self._current_action = None

                    self._current_chunk_start_global = global_frame
                    self._current_chunk_length = 0
                    ModelVars.current_anim_name = anim_name

                local_frame = int(global_frame - self._current_chunk_start_global)

                if self._current_action is not None:
                    # fast write vertex coords
                    self._mesh.vertices.foreach_set('co', flatten_extend(ModelVars.all_verts[frame_index]))

                    # ensure OBJECT mode for safe operations
                    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
                        bpy.ops.object.mode_set(mode='OBJECT')

                    # insert keyframes into the mesh-level action fcurves
                    for idx, v in enumerate(self._mesh.vertices):
                        v.co = ModelVars.all_verts[frame_index][idx]
                        self._insert_vertex_into_action(self._current_action, idx, v.co, local_frame)

                self._current_chunk_length = max(self._current_chunk_length, local_frame)

                self._frame_index += 1
                processed += 1

            # update progress
            context.window_manager.progress = self._frame_index / self._frame_count
            context.window_manager.progress_message = "Animations:"

            # Update UI
            for area in context.screen.areas:
                area.tag_redraw()

            # finished?
            if self._frame_index >= self._frame_count:
                # finalize last chunk
                if self._current_action is not None:
                    self._finalize_current_action()

                # unmute only the first track, mute the rest
                for i, t in enumerate(self._created_tracks):
                    t.mute = (i != 0)

                # restore mode
                if self._orig_mode and bpy.context.object and bpy.context.object.mode != self._orig_mode:
                    bpy.ops.object.mode_set(mode=self._orig_mode)

                # set scene range to longest action length
                max_len = 0
                for t in self._created_tracks:
                    for s in t.strips:
                        max_len = max(max_len, int(s.frame_end))
                if max_len > 0:
                    bpy.context.scene.frame_start = 0
                    bpy.context.scene.frame_end = max_len

                # end progress and cleanup
                context.window_manager.progress = 0.0
                context.window_manager.progress_message = ""
                wm.event_timer_remove(self._timer)
                self._timer = None

                return {'FINISHED'}

        # allow user to cancel with ESC
        if event.type in {'ESC'}:
            if self._timer:
                wm.event_timer_remove(self._timer)
                self._timer = None

            context.window_manager.progress = 0.0
            context.window_manager.progress_message = ""

            # Update UI
            for area in context.screen.areas:
                area.tag_redraw()

            self.report({'CANCELLED'})
            return {'CANCELLED'}

        return {'PASS_THROUGH'}





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

        # Update UI
        for area in context.screen.areas:
            area.tag_redraw()

        return {'FINISHED'}







