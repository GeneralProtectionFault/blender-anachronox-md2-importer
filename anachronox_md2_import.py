bl_info = {
    "name": "Anachronox: Experimental MD2 Importer",
    "author": "Lennart G, Alpaca, Holonet, Creaper",
    "location": "File > Import > Anachronox (.md2)",
    "version": (0, 1, 3),
    "blender": (2, 80, 0),
    "category": "Import-Export"
}

import sys
from dataclasses import dataclass, fields
import struct
from pathlib import Path
from typing import List
"""
This part is used to load an md2 file into a MD2 dataclass object
"""
""" 
Dataclasses resembling structs in C. Used for storing MD2 information, being nested and forming one big dataclass
"""

@dataclass
class vec3_t:
    x: float
    y: float
    z: float


@dataclass
class vertex_t:  # 4 bytes in total
    v: list  # unsigned char (in python 1 byte int), list of len 3, compressed vertex
    lightnormalindex: int  # unsigned char, index to a normal vector for the lighting


@dataclass
class frame_t:  # 40 + num_xyz*4 bytes
    scale: vec3_t  # scale values, 3 elements
    translate: vec3_t  # translation vector, 3 elements
    name: str  # frame name, 16 characters aka bytes at most
    verts: List[vertex_t]  # list of num_xyz vertex_t's


@dataclass
class md2_t:
    ident: int                          # magic number. must be equal to "IDP2" or 844121161 as int
    version: int                        # md2 version. must be equal to 15
    resolution: int                     # Vertex resolution flag 0=3, 1=4, 2=6

    skinwidth: int                      # width of the texture
    skinheight: int                     # height of the texture
    framesize: int                      # size of one frame in bytes

    num_skins: int                      # number of textures
    num_xyz: int                        # number of vertices
    num_st: int                         # number of texture coordinates
    num_tris: int                       # number of triangles
    num_glcmds: int                     # number of opengl commands
    num_frames: int                     # total number of frames

    ofs_skins: int                      # offset to skin names (64 bytes each)
    ofs_st: int                         # offset to s-t texture coordinates
    ofs_tris: int                       # offset to triangles
    ofs_frames: int                     # offset to frame data
    ofs_glcmds: int                     # offset to opengl commands
    ofs_end: int                        # offset to end of file
    num_header_ofs_dec_68: int          # unknown number specified in the header at 44H
    ofs_end_fan: int                    # offset to the end of 
    num_LODdata1: int                   # Float LOD data 1
    num_LODdata2: int                   # Float LOD data 2
    num_LODdata3: int                   # Float LOD data 3
    num_tsurf: int                      # number of Tagged Surfaces
    ofs_tsurf: int                      # offset to Tagged surfaces 
    


@dataclass
class triangle_t:  # 12 bytes each
    vertexIndices: List[int]  # short, 3 values
    textureIndices: List[int]  # short, 3 values


@dataclass
class textureCoordinate_t: # 4 bytes each
    s: int  # short
    t: int  # short


@dataclass
class glCommandVertex_t:
    s: float
    t: float
    vertexIndex: int


@dataclass
class glCommand_t:
    mode: str  # string saying GL_TRIANGLE_STRIP or GL_TRIANGLE_FAN
    vertices: List[glCommandVertex_t]  # all vertices rendered with said mode


@dataclass
class md2_object:
    header: md2_t
    skin_names: List[str]
    triangles: List[triangle_t]
    frames: List[frame_t]
    texture_coordinates: List[textureCoordinate_t]
    gl_commands: List[glCommand_t]
    tsurf_names: List[str]


"""
Functions used to create an MD2 Object
"""
def load_gl_commands(gl_command_bytes):
    """
    Loads gl_commands which are a list of GL_TRIANGLE_STRIP and GL_TRIANGLE_FAN calls that reduce fps
    Code differs much from original loading code in C
    :param gl_command_bytes: bytes belonging to gl_commands lump from md2 file
    :return: list of dataclasses storing gl commands
    """
    offset = 0
    gl_commands = list()
    while True:  # ends when mode is 0
        (mode,) = struct.unpack("<i", gl_command_bytes[offset:offset+4])
        num_verts = abs(mode)
        if mode > 0:
            mode = "GL_TRIANGLE_STRIP"
        elif mode == 0:
            offset += 4
            break
        else:
            mode = "GL_TRIANGLE_FAN"
        offset += 4
        gl_vertices = list()
        for i in range(num_verts):
            s_and_t = struct.unpack("<ff", gl_command_bytes[offset+12*i:offset+12*i+8])
            vertex_index = struct.unpack("<i", gl_command_bytes[offset+12*i+8:offset+12*i+12])
            gl_vertices.append(glCommandVertex_t(*s_and_t, *vertex_index))
        # print(gl_vertices)
        offset += 12*num_verts
        gl_commands.append(glCommand_t(mode, gl_vertices))
    return gl_commands



def load_triangles(triangle_bytes, header):
    """
    Creates basic list of triangle dataclasses which contain indices to vertices
    :param triangle_bytes: bytes from md2 file belonging to triangles lump
    :param header: dataclass containing header information
    :return: list of triangles
    """
    triangles = list()
    for i in range(header.num_tris):
        triangle = triangle_t(list(struct.unpack("<hhh", triangle_bytes[12*i:12*i+6])), list(struct.unpack("<hhh", triangle_bytes[12*i+6:12*i+12])))
        # print(triangle)
        triangles.append(triangle)
    return triangles


def load_frames(frames_bytes, header):
    """
    Loads frames
    :param frames_bytes: bytes from md2 file belonging to frames lump
    :param header: header dataclass
    :return: list of frame dataclass objects
    """
    # # check if header.ofs_glcmds - header.ofs_frames == header.num_frames*(40+4*header.num_xyz) # #
    #print("len", len(frames_bytes))
    #print("frames", header.num_frames)
    #print("check", header.num_frames*(40+4*header.num_xyz))
    frames = list()
    frame_names = list()
    if header.resolution == 0: #vertex information in 3 bytes (8 bit vertices)
        unpack_format = "<BBB"
        resolution_bytes = 0
    elif header.resolution == 1: #vertex information in 4 bytes (11 bit X, 10 bit Y, 11 bit Z)
        unpack_format = "<BBB"
        resolution_bytes = 1
    elif header.resolution == 2: #vertex information in 6 bytes (16 bit vertices)
        unpack_format = "<HHH"
        resolution_bytes = 3

    for current_frame in range(header.num_frames):
        scale = vec3_t(*struct.unpack("<fff", frames_bytes[(40+(5+resolution_bytes)*header.num_xyz)*current_frame:(40+(5+resolution_bytes)*header.num_xyz)*current_frame+12]))
        translate = vec3_t(*struct.unpack("<fff", frames_bytes[(40+(5+resolution_bytes)*header.num_xyz)*current_frame+12:(40+(5+resolution_bytes)*header.num_xyz)*current_frame+24]))
        name = frames_bytes[(40+(5+resolution_bytes)*header.num_xyz)*current_frame+24:(40+(5+resolution_bytes)*header.num_xyz)*current_frame+40].decode("ascii", "ignore")
        #print("name", name)
        verts = list()
        for v in range(header.num_xyz):
            #print(v)
            if header.resolution == 0 or header.resolution == 2:
                verts.append(vertex_t(list(struct.unpack(unpack_format, frames_bytes[(40+(5+resolution_bytes)*header.num_xyz)*current_frame+40+v*(5+resolution_bytes):(40+(5+resolution_bytes)*header.num_xyz)*current_frame+40+v*(5+resolution_bytes)+(3+resolution_bytes)])), *struct.unpack("<H", frames_bytes[(40+(5+resolution_bytes)*header.num_xyz)*current_frame+(43+resolution_bytes)+v*(5+resolution_bytes):(40+(5+resolution_bytes)*header.num_xyz)*current_frame+(45+resolution_bytes)+v*(5+resolution_bytes)])))  # list() only for matching expected type
            elif header.resolution == 1:
                vertexBytes = int.from_bytes(frames_bytes[(40+6*header.num_xyz)*current_frame+40+v*6:(40+6*header.num_xyz)*current_frame+40+v*6+4], sys.byteorder)
                x = vertexBytes >> 0 & 0x000007ff
                y = vertexBytes >> 11 & 0x000003ff
                z = vertexBytes >> 21 & 0x000007ff
                vector = [x,y,z]
                normal = struct.unpack("<H", frames_bytes[(40+6*header.num_xyz)*current_frame+44+v*6:(40+6*header.num_xyz)*current_frame+46+v*6])[0]
                vertex = vertex_t(vector, normal)
                verts.append(vertex)
        name = name.rstrip("\x00")
        frame_names.append(name)
        #print(scale, translate, name, verts)
        frame = frame_t(scale, translate, name, verts)
        #print("Frame: ",frame)
        frames.append(frame)
        #print("Frame names ", frame_names)
#    print("Frame Names", frame_names) # write code to count the number of frames in each frame name
    return frames


def load_header(file_bytes):
    """
    Creates header dataclass object
    :param file_bytes: bytes from md2 file belonging to header
    :return: header dataclass object
    """
    # print(file_bytes[:4].decode("ascii", "ignore"))
    arguments = struct.unpack("<ihhiiiiiiiiiiiiiiiiifffii", file_bytes[:96])
    header = md2_t(*arguments)
    # Verify MD2
    if not header.ident == 844121161 or not header.version == 15:
        print(f"Error: File type is not MD2. Ident or version not matching")
        print(f'Ident: {file_bytes[:4].decode("ascii", "ignore")} should be "IDP2"')
        print(f"Version: {header.version} should be 15")


    print("--------------- HEADER VALUES -------------------")
    for field in fields(header):
        print(f"{field.name} - ", getattr(header, field.name))

    print("--------------------------------------------------")

    return header


def load_texture_coordinates(texture_coordinate_bytes, header):
    """
    Loads UV (in Q2 term ST) coordinates
    :param texture_coordinate_bytes:
    :param header:
    :return: list of texture coordinate dataclass objects
    """
    texture_coordinates = list()
    for i in range(header.num_st):
        texture_coordinates.append(textureCoordinate_t(*struct.unpack("<hh", texture_coordinate_bytes[4*i:4*i+4])))
    return texture_coordinates


def load_file(path):
    """
    Master function returning one dataclass object containing all the MD2 information
    :param path:
    :return:
    """
    with open(path, "rb") as f:  # bsps are binary files
        byte_list = f.read()  # stores all bytes in bytes1 variable (named like that to not interfere with builtin names
    header = load_header(byte_list)
    skin_names = [byte_list[header.ofs_skins + 64 * x:header.ofs_skins + 64 * x + 64].decode("ascii", "ignore") for x in range(header.num_skins)]
    #print("Skin Names ", skin_names)
    tsurf_names = [byte_list[header.ofs_tsurf + 12 * x:header.ofs_tsurf + 12 * x + 8].decode("ascii", "ignore") for x in range(header.num_tsurf)]
    #print("Tagged Surface Names ", tsurf_names)
    #todo Grab number of tagged triangles per tagged surface and the data right before the first tagged surface name that I have momentarily forgotten what it is
    

    triangles = load_triangles(byte_list[header.ofs_tris:header.ofs_frames], header)
    frames = load_frames(byte_list[header.ofs_frames:header.ofs_glcmds], header)
    texture_coordinates = load_texture_coordinates(byte_list[header.ofs_st:header.ofs_tris], header)
    gl_commands = load_gl_commands(byte_list[header.ofs_glcmds:header.ofs_end_fan])
    
    # print(header)
    # print(skin_names)
    # print(triangles)
    # print(frames)
    #print("texture coords ")
    #print(texture_coordinates)
    # UV coordinates not correctly using 0,1 space, so I temporarily put in this hack to fix it - Creaper
    header.skinwidth_adjusted = header.skinwidth
    header.skinheight_adjusted = header.skinheight
    for i in range(len(texture_coordinates)):
        # texture_coordinates[i].s = texture_coordinates[i].s+2
        # print("texture_coordinates[i].s ")
        # print(texture_coordinates[i].s)
        # print("skin width ")
        # print(header.skinwidth)
        if header.resolution == 0: # UV scaling fix for resolution 0 only for first texture
            texture_coordinates[i].s = (texture_coordinates[i].s) /header.skinwidth_adjusted*1.0667
            texture_coordinates[i].t = texture_coordinates[i].t / header.skinheight_adjusted*1.0322
        elif header.resolution == 1: # UV scaling fix for resolition 1 only for first texture
            texture_coordinates[i].s = (texture_coordinates[i].s) /header.skinwidth_adjusted*1.0323
            texture_coordinates[i].t = texture_coordinates[i].t / header.skinheight_adjusted*1.0159
        elif header.resolution == 2: # UV scaling fix for resolution 2 only for first texture
            texture_coordinates[i].s = (texture_coordinates[i].s) /header.skinwidth_adjusted*1.0156
            texture_coordinates[i].t = texture_coordinates[i].t / header.skinheight_adjusted*1.0077
            
        # print("texture_coordinates[i].s after ")
        # print(texture_coordinates[i].s)
        # print(f"Coordinate {i} - {texture_coordinates[i]}")
    
    print(f"Number of vertices: {header.num_xyz}\n")
    
    for i_frame in range(len(frames)):
        for i_vert in range((header.num_xyz)):
            frames[i_frame].verts[i_vert].v[0] = frames[i_frame].verts[i_vert].v[0]*frames[i_frame].scale.x+frames[i_frame].translate.x
            frames[i_frame].verts[i_vert].v[1] = frames[i_frame].verts[i_vert].v[1] * frames[i_frame].scale.y + frames[i_frame].translate.y
            frames[i_frame].verts[i_vert].v[2] = frames[i_frame].verts[i_vert].v[2] * frames[i_frame].scale.z + frames[i_frame].translate.z
    model = md2_object(header, skin_names, triangles, frames, texture_coordinates, gl_commands, tsurf_names)
    return model


import bpy
import sys
from importlib import reload # required when a self-written module is imported that's edited simultaneously
import os  # for checking if skin pathes exist


def blender_load_md2(md2_path, displayed_name, model_scale):
    """
    This function uses the information from a md2 dataclass into a blender object.
    This will consist of an animated mesh and its material (which is not much more than the texture.
    For better understanding, steps are:
        - Create the MD2 object containing all information that's inside the loaded md2
        - Get the absolute path of the UV map / skin to load
        - Get necessary information about the mesh (vertices, tris, uv coordinates)
        - Create the scene structure and create the mesh for the first frame
        - Assign UV coordinates to each triangle
        - Create shape animation (Add keyframe to each vertex)
        - Assign skin to mesh
    """
    """ Create MD2 dataclass object """
    # ImageFile.LOAD_TRUNCATED_IMAGES = True # Necessary for loading jpgs with PIL
    print("md2_path: ", md2_path)
    #object_path = md2_path  # Kept for testing purposes
    model_path = '\\'.join(md2_path.split('\\')[0:-1])+"\\"
    print("Model Path: ", model_path)
    model_filename = "\\".join(md2_path.split("\\")[-1:])
    print("Model Filename: ", model_filename)

    # A dataclass containing all information stored in a .md2 file
    my_object = load_file(md2_path)


    """ Create skin path. By default, the one stored inside of the MD2 is used. Some engines like the Digital Paintball 2 one
    check for any image file with that path disregarding the file extension.
    """
    """ get the skin path stored inside of the MD2 """
    # check box must be checked (alternatively it could be checked if the input field was empty or not ...)
    texture_paths = []
    my_object.skin_resolutions = {}

    for index, skin_name in enumerate(my_object.skin_names):
        embedded_texture_name = my_object.skin_names[index].rstrip("\x00")
        embedded_texture_name_unextended = os.path.splitext(embedded_texture_name)[0] # remove extension (last one)
        print("Embedded Texture Name " +embedded_texture_name)
        """ Look for existing file of given name and supported image format """
        supported_image_formats = [".png", ".jpg", ".jpeg", ".bmp", ".pcx", ".tga"] # Order doesn't match DP2 image order
        for format in supported_image_formats:
            # Added support for autoloading textures and to name mesh the same as filename if a Display Name is not entered on import screen - Creaper
            texture_path = model_path + embedded_texture_name_unextended + format
            if os.path.isfile(texture_path):
                print("Texture found: " + texture_path)

                # Get the resolution from the actual image while we're here, as the header only has the first one, which won't cut it for multi-textured models - Holonet
                with PIL.Image.open(texture_path) as img:
                    width, height = img.size
                    my_object.skin_resolutions[embedded_texture_name] = img.size
                    print(f"Texture resolution, {embedded_texture_name}: {my_object.skin_resolutions[embedded_texture_name]}")
                
                break
            else:
                print("Unable to locate texture " +model_path+embedded_texture_name_unextended+format +"!")
        texture_paths.append(texture_path)
        
    """ Loads required information for mesh generation and UV mapping from the .md2 file"""
    # Gets name to give to the object and mesh in the outliner
    if not displayed_name:
        # Added support for autoloading textures and to name mesh the same as filename if a Display Name is not entered on import screen - Creaper
        # object_name_test = "/".join(object_path.split("/")[-2:]).split(".")[:-1]
        #   
        object_name = os.path.basename(md2_path).split('/')[-1]
        object_name = os.path.splitext(object_name)[0] # remove extension (last one)
        print("Blender Outliner Object Name: ", object_name)
        mesh = bpy.data.meshes.new(object_name)  # add the new mesh via filename
        

    else:
        object_name = [displayed_name]
        print("Blender Outliner Object Name: ", displayed_name)
        mesh = bpy.data.meshes.new(*object_name)  # add the new mesh, * extracts string from Display Name input list


    # List of vertices [x,y,z] for all frames extracted from the md2 object
    all_verts = [[x.v for x in my_object.frames[y].verts] for y in range(my_object.header.num_frames)]
    # List of vertex indices forming a triangular face
    tris = ([x.vertexIndices for x in my_object.triangles])
    # uv coordinates (in q2 terms st coordinates) for projecting the skin on the model's faces
    # blender flips images upside down when loading so v = 1-t for blender imported images
    uvs_others = ([(x.s, 1-x.t) for x in my_object.texture_coordinates]) 
    # blender uv coordinate system originates at lower left

    """ Lots of code (copy and pasted) that creates a mesh and adds it to the scene collection/outlines """
    obj = bpy.data.objects.new(mesh.name, mesh)
    col = bpy.data.collections.get("Collection")
    col.objects.link(obj)
    bpy.context.view_layer.objects.active = obj

    # Creates mesh by taking first frame's vertices and connects them via indices in tris
    mesh.from_pydata(all_verts[0], [], tris) 

    """ UV Mapping: Create UV Layer, assign UV coordinates from md2 files for each face to each face's vertices """
    uv_layer=(mesh.uv_layers.new())
    mesh.uv_layers.active = uv_layer
    # add uv coordinates to each polygon (here: triangle since md2 only stores vertices and triangles)
    # note: faces and vertices are stored exactly in the order they were added
    for face_idx, face in enumerate(mesh.polygons):
        for idx, (vert_idx, loop_idx) in enumerate(zip(face.vertices, face.loop_indices)):
            uv_layer.data[loop_idx].uv = uvs_others[my_object.triangles[face_idx].textureIndices[idx]]
                
    """ Create animation for animated models: set keyframe for each vertex in each frame individually """
    # Create keyframes from first to last frame
    for i in range(my_object.header.num_frames):
        for idx,v in enumerate(obj.data.vertices):
            obj.data.vertices[idx].co = all_verts[i][idx]
            v.keyframe_insert('co', frame=i*2)  # parameter index=2 restricts keyframe to dimension

    # insert first keyframe after last one to yield cyclic animation
    # for idx,v in enumerate(obj.data.vertices):
    # 	obj.data.vertices[idx].co = all_verts[0][idx]
    # 	v.keyframe_insert('co', frame=60)

 


    """ Assign skin to mesh: Create material (barely understood copy and paste again) and set the image. 
    Might work by manually setting the textures pixels to the pixels of a PIL.Image if it would actually
    load non-empty .pcx files
    idea/TODO: Write an own pcx loader from scratch ... """
    texpath=0
    for nameindex, tsurf_name in enumerate(my_object.tsurf_names):
        tagged_surface_name = my_object.tsurf_names[nameindex].rstrip("\x00")
        print("Embedded Tagged Surface Name: " +tagged_surface_name)
        material_name = ("M_" + tagged_surface_name)#"\\".join(texture_path_unextended.split("\\")[-1:]))
        print("Blender Material name: " + material_name)
        mat = bpy.data.materials.new(name=material_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs['Specular'].default_value = 0
        texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
        # Changed texture names to correspond with Tagged Surface names proceeded with 'M_'
        #texture_path_unextended = os.path.splitext(texture_path)[0] # remove extension (last one)
        if(texpath < len(texture_paths)):
            print("Material Texture: ", texture_paths[texpath])
            texture_path = texture_paths[texpath]
            path = Path(texture_path)
            if path.exists():
                texImage.image = bpy.data.images.load(texture_paths[texpath])
                # again copy and paste
            else:
                print("Cannot find texture " + texture_paths[i] + "!")
                print("Check " +model_path+ " for .mda material texture file.")
                texImage.image = bpy.data.images.load(texture_paths[texpath])
                print("Assigning texture " + texture_paths[texpath])
        else:
            texImage.image = bpy.data.images.load(texture_paths[0])
            print("Material Texture: ", texture_paths[0])

        texpath = texpath +1
        mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
        obj.data.materials.append(mat)

        # Added support to resize the model to the desired scale input at import screen
        print("New model scale: ", model_scale)
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
        
        # bpy.data.objects[obj_name].scale = (model_scale, model_scale, model_scale)
        obj.scale = (model_scale, model_scale, model_scale)
        # Apply the scale to normalize the object after adjusting scale
        # Doesn't work, but works fine in Blender scripting tab.  Race condition?
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        


    print("YAY NO ERRORS!!")
    return {'FINISHED'} # no idea, seems to be necessary for the UI
        
"""
This part is required for the UI, to make the Addon appear under File > Import once it's
activated and to have additional input fields in the file picking menu
Code is taken from Templates > Python > Operator File Import in Text Editor
The code here calls blender_load_md2
"""

# ImportHelper is a helper class, defines filename and
# invoke() function which calls the file selector.
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, BoolProperty, EnumProperty
from bpy.types import Operator


class ImportSomeData(Operator, ImportHelper):
    """Loads a Quake 2 MD2 File"""
    bl_idname = "import_md2.some_data"  # important since its how bpy.ops.import_test.some_data is constructed
    bl_label = "Import MD2"

    ## ImportHelper mixin class uses this
    #filename_ext = ".md2"

    filter_glob: StringProperty(
        default="*.md2", # only shows md2 files in opening screen
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )
    
    # List of operator properties, the attributes will be assigned
    # to the class instance from the operator settings before calling.
    displayed_name: bpy.props.StringProperty(name="Outliner name",
                                        description="Desired model name in the outliner.\nGood for renaming model to coincide with entity.dat.",
                                        default="",
                                        maxlen=1024)
    # Added support to resize the model to the desired scale input at import screen
    model_scale: bpy.props.FloatProperty(name="New Scale",
                                        description="Desired scale for the model.\nGood for recaling the model to fit other scale systems. I.E. .0254 is the scale for Unreal Engine.",
                                        default=.0254)
                                        
    
    def execute(self, context):
        return blender_load_md2(self.filepath, self.displayed_name, self.model_scale)


# Only needed if you want to add into a dynamic menu
def menu_func_import(self, context):
    self.layout.operator(ImportSomeData.bl_idname, text="Anachronox Model Import (.md2)")

# called when addon is activated (adds script to File > Import
def register():
    import subprocess
    import sys
    import os
    
    # path to python.exe
    python_exe = os.path.join(sys.prefix, 'bin', 'python.exe')
    
    # upgrade pip
    subprocess.call([python_exe, "-m", "ensurepip"])
    subprocess.call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
    
    # install required packages
    subprocess.call([python_exe, "-m", "pip", "install", "pillow"])

    bpy.utils.register_class(ImportSomeData)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)

# called when addon is deactivated (removed script from menu)
def unregister():
    bpy.utils.unregister_class(ImportSomeData)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)


if __name__ == "__main__":
    
    register()

    # test call
    bpy.ops.import_md2.some_data('INVOKE_DEFAULT')
