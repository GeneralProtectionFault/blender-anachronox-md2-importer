import bpy
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

        bpy.ops.object.mode_set(mode = 'EDIT')

        frame_count = len(ModelVars.my_object.frames)

        frame_index_list = list()
        # frame_x_values = list()
        # frame_y_values = list()
        # frame_z_values = list()
        for frame_index, frame in enumerate(ModelVars.my_object.frames): 
            frame_index_list.append(frame_index * 2)
        #     frame_x_values.append(ModelVars.all_verts[frame_index][0])
        #     frame_y_values.append(ModelVars.all_verts[frame_index][1])
        #     frame_z_values.append(ModelVars.all_verts[frame_index][2])
        
        # frame_co_values = list()
        # frame_co_values.append(frame_x_values)
        # frame_co_values.append(frame_y_values)
        # frame_co_values.append(frame_z_values)

        # Sets the end frame on the timeline to what will be the last frame
        bpy.context.scene.frame_end = frame_index_list[-1]

        ModelVars.obj.animation_data_create()
       
        print("Looping through frames...")
        for frame_index, frame in enumerate(ModelVars.my_object.frames):
        # for idx, v in enumerate(ModelVars.obj.data.vertices):
            # Update progress every 10 frames
            if (frame_index % 10 == 0):
                showProgress(frame_index, frame_count)

            action = ModelVars.obj.animation_data.action = bpy.data.actions.new(name=f"{ModelVars.object_name}")

            # fcurve_x = action.fcurves.new(f"vertices", index = 0, action_group = "X Position (Mesh Vertex)")
            
            # fcurve_x.keyframe_points.add(count=len(ModelVars.my_object.frames))
            # # fcurve_x.keyframe_points.foreach_set("co", \
            # # [x for co in zip(frame_index_list, flatten_extend(ModelVars.all_verts[frame_index])) for x in co])
            # # [x for co in zip(frame_index_list, frame_x_values) for x in co])

            # fcurve_x.sampled_points.foreach_set("co", \
            # [x for co in zip(frame_index_list, flatten_extend(ModelVars.all_verts[frame_index])) for x in co])
            
            # fcurve_x.update()





            anim_name = frame.name[ : findnth(frame.name, '_', 2)]
            # Indicates we're on the next animation!
            # if anim_name != ModelVars.current_anim_name:
            #     ModelVars.current_anim_name = anim_name
            #     ModelVars.animation_list.append(anim_name)

            #     # Create a new action for the new animation
            #     # bpy.ops.nla.actionclip_add(action=f"{ModelVars.object_name}_{anim_name}")
            #     # print(f"New action created: {ModelVars.object_name}_{anim_name}")
            
            # data_path = "vertices[%d].co"
            # data_path = "vertices[%d]"

            # fcurves = [action.fcurves.new(data_path % v.index, index =  i) for i in range(3)]    
            # for n, fcurve in enumerate(fcurves):
            #     fcurve.keyframe_points.add(count=len(ModelVars.my_object.frames))
                
            #     #fcurve.keyframe_points.insert(frame_index_list[idx], frame_co_values[idx], options={'FAST'})
                
            #     fcurve.keyframe_points.foreach_set("co", \
            #     [x for co in zip(frame_index_list, flatten_extend(frame_co_values[n])) for x in co])



            # for frm, value in zip(frame_index_list, flatten_extend(ModelVars.all_verts[frame_index])):
            #     insert_keyframe(fcurves, frm, value)   
            


            ModelVars.obj.data.vertices.foreach_set('co', flatten_extend(ModelVars.all_verts[frame_index]))

            for idx, v in enumerate(ModelVars.obj.data.vertices):
                ModelVars.obj.data.vertices[idx].co = ModelVars.all_verts[frame_index][idx]
                # parameter index = 0, 1 or 2 restricts to x, y or z axis
                success = v.keyframe_insert('co', frame = frame_index * 2) # Back in the olden days, we only had them thar 30 frames/second
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
        
        # REMOVED - This doesn't work on all frames ********************************
        # Any rotation needs to be applied to all frames
        # for frame_index, frame in enumerate(ModelVars.my_object.frames):
        #     bpy.context.scene.frame_set(frame_index)
        #     bpy.context.active_object.rotation_euler[0] = math.radians(ModelVars.x_rotate) # rotate on import axis=(1=X 2=Y, 3=Z) degrees=(amount)
        #     bpy.context.active_object.rotation_euler[1] = math.radians(ModelVars.y_rotate) # rotate on import axis=(1=X 2=Y, 3=Z) degrees=(amount)
        #     bpy.context.active_object.rotation_euler[2] = math.radians(ModelVars.z_rotate) # rotate on import axis=(1=X 2=Y, 3=Z) degrees=(amount)

        # Reset
        bpy.context.scene.frame_set(0)
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
  






