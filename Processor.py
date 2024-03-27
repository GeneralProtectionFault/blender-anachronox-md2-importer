import bpy
from dataclasses import dataclass, fields
from typing import List
import math

from .utils import ModelVars
from .utils import startProgress, showProgress, endProgress, findnth


class QueueRunner(bpy.types.Macro):
    """
    This macro will fire the operators in this file off in order to keep from locking things up.
    (The frames take a bit of time to process)
    """
    bl_idname = "amd2_import.macro"
    bl_label = "Anachronox MD2 Import Macro"


class ImportAnimationFrames(bpy.types.Operator):
    bl_idname = "wm.import_animation_frames"
    bl_label = "Import Animation Frames"

    def execute(self, context):
        self.finished = False
        frame_count = len(ModelVars.my_object.frames)

        bpy.ops.object.mode_set(mode = 'EDIT')

        print("Looping through frames...")

        for frame_index, frame in enumerate(ModelVars.my_object.frames):
            # Update progress every 10 frames
            if (frame_index % 10 == 0):
                showProgress(frame_index, frame_count)

            anim_name = frame.name[ : findnth(frame.name, '_', 2)]
            # print(f"Frame {frame_index} name: {anim_name}")

            # Indicates we're on the next animation!
            if anim_name != ModelVars.current_anim_name:
                ModelVars.current_anim_name = anim_name
                ModelVars.animation_list.append(anim_name)

                # Create a new action for the new animation
                # bpy.ops.nla.actionclip_add(action=f"{ModelVars.object_name}_{anim_name}")
                # print(f"New action created: {ModelVars.object_name}_{anim_name}")

            for idx,v in enumerate(ModelVars.obj.data.vertices):
                ModelVars.obj.data.vertices[idx].co = ModelVars.all_verts[frame_index][idx]
                success = v.keyframe_insert('co', frame=frame_index*2)  # parameter index=2 restricts keyframe to dimension
                if not success:
                    print(f'Keyframe insert failed, Frame: {frame_index}')

        bpy.ops.object.mode_set(mode = 'OBJECT')
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
                print(f"Cannot find textures for {ModelVars.md2_path}!")
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
        # bpy.data.objects[obj_name].scale = (model_scale, model_scale, model_scale)
        
        # ***** REMOVED ****** This doesn't really work because it doesn't hit all the frames on animated models, done in the frames instead (scale & translate)
        # obj.scale = (model_scale, model_scale, model_scale)
        # print(f"New model scale: { model_scale}")
        # print("New model scale applied...")
        bpy.context.active_object.rotation_euler[0] = math.radians(ModelVars.x_rotate) # rotate on import axis=(1=X 2=Y, 3=Z) degrees=(amount)
        bpy.context.active_object.rotation_euler[1] = math.radians(ModelVars.y_rotate) # rotate on import axis=(1=X 2=Y, 3=Z) degrees=(amount)
        bpy.context.active_object.rotation_euler[2] = math.radians(ModelVars.z_rotate) # rotate on import axis=(1=X 2=Y, 3=Z) degrees=(amount)
        print("Object rotated per selected parameters...")

        # Apply Transforms if option selected on import screen
        if(ModelVars.apply_transforms):
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
        if(ModelVars.recalc_normals):
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
  






