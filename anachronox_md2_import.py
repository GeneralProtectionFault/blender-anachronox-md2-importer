import bpy
import os
import sys
import subprocess
from dataclasses import dataclass, fields
import struct
from pathlib import Path
from typing import List
import platform

from .utils import startProgress, showProgress, endProgress, ModelVars, findnth

from importlib import reload # required when a self-written module is imported that's edited simultaneously
import math # for applying optional rotate on import
# import mathutils
# from mathutils import Vector

from .Processor import QueueRunner


# path to python.exe
if platform.system() == "Linux":
    python_exe = os.path.join(sys.prefix, 'bin', 'python')
else:
    python_exe = os.path.join(sys.prefix, 'bin', 'python.exe')

try:
    # upgrade pip
    subprocess.call([python_exe, "-m", "ensurepip"])
    
    # This doesn't jive well with Blender's Python environment for whatever reason...
    # subprocess.call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
except Exception as argument:
    print(f"Issue ensuring/upgrading pip:\n{argument}")


# install required packages
try:
    subprocess.call([python_exe, "-m", "pip", "install", "pillow"])
    # subprocess.call([python_exe, "-m", "pip", "install", "mathutils"])

except ImportError as argument:
    print(f"ERROR: Pillow/PIL failed to install\n{argument}")



import PIL
from PIL import Image, ImagePath


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
class vertex_indexed: # Store the vertex order for matching to triangle data
    index: int
    v: list


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
    ofs_end_fan: int                    # offset to the end of the gl commands 
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
    tagged_surfaces: dict()
    vertices: List[vertex_indexed]
    triangle_skin_dict: dict()
    extra_data: dict()
    texture_paths: list()


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
        (mode,) = struct.unpack("<i", gl_command_bytes[offset : offset + 4]) # 4 bytes - 1 int
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
            s_and_t = struct.unpack("<ff", gl_command_bytes[offset + 12 * i : offset + 12 * i + 8]) # 8 bytes - 2 floats
            vertex_index = struct.unpack("<i", gl_command_bytes[offset + 12 * i + 8 : offset + 12 * i + 12]) # 4 bytes - 1 int
            gl_vertices.append(glCommandVertex_t(*s_and_t, *vertex_index))
        # print(gl_vertices)
        
        offset += 12 * num_verts
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


def load_frames(frames_bytes, header, model_scale):
    """
    Loads frames
    :param frames_bytes: bytes from md2 file belonging to frames lump
    :param header: header dataclass
    :return: list of frame dataclass objects
    """
    # # check if header.ofs_glcmds - header.ofs_frames == header.num_frames*(40+4*header.num_xyz) # #
    # print("len", len(frames_bytes))
    # print("frames", header.num_frames)
    # print("check", header.num_frames*(40+4*header.num_xyz))
    frames = list()
    frame_names = list()
    if header.resolution == 0: # vertex information in 3 bytes (8 bit vertices)
        unpack_format = "<BBB"
        resolution_bytes = 0
    elif header.resolution == 1: # vertex information in 4 bytes (11 bit X, 10 bit Y, 11 bit Z)
        unpack_format = "<BBB"
        resolution_bytes = 1
    elif header.resolution == 2: # vertex information in 6 bytes (16 bit vertices)
        unpack_format = "<HHH"
        resolution_bytes = 3

    for current_frame_number in range(header.num_frames):
        
        frame_start = (40 + (5 + resolution_bytes) * header.num_xyz) * current_frame_number  

        # Get any scaling for this frame of animation
        scale = vec3_t(*struct.unpack("<fff", frames_bytes[frame_start : frame_start + 12]))
        
        
        # print(scale)
        # Get any movement for this frame of animation
        translate = vec3_t(*struct.unpack("<fff", frames_bytes[frame_start + 12 : frame_start + 24]))

        # print(f"Scale Before: {scale}")
        # print(f"Translate Before: {translate}")

        ##### This applies the scaling set in the UI when importing ##############################################
        scale = vec3_t(x = scale.x * model_scale, y = scale.y * model_scale, z = scale.z * model_scale)
        translate = vec3_t(translate.x * model_scale, translate.y * model_scale, translate.z * model_scale)
        ##########################################################################################################

        # print(f"Scale After: {scale}")
        # print(f"Translate After: {translate}")

        # Animation name
        name = frames_bytes[frame_start + 24 : frame_start + 40].decode("ascii", "ignore")

        verts = list()
        # print(f"Vertex Resolution: {header.resolution}")

        # Loop for the number of vertices
        for v in range(header.num_xyz):
            if header.resolution == 0 or header.resolution == 2:
                # First mess is the vertex (vector)
                vertex_start_index = frame_start + 40 + v * (5 + resolution_bytes)
                vertex_end_index = vertex_start_index + (3 + resolution_bytes)
                # print(f"Vertex index range: {start_index}, {end_index}")
                lightnormal_start_index = frame_start + (43 + resolution_bytes) + v * (5 + resolution_bytes)
                lightnormal_end_index = frame_start + (45 + resolution_bytes) + v * (5 + resolution_bytes)

                # struct.unpack returns tuple--in this case, 3 bytes, which are coordinates for the vertex
                vector = list(struct.unpack(unpack_format, frames_bytes[vertex_start_index : vertex_end_index])) # list() only for matching expected typ
                normal = struct.unpack("<H", frames_bytes[lightnormal_start_index : lightnormal_end_index])


            elif header.resolution == 1:
                vertex_start_index = (40 + 6 * header.num_xyz) * current_frame_number + 40 + v * 6
                vertexBytes = int.from_bytes(frames_bytes[vertex_start_index : vertex_start_index + 4], sys.byteorder)
                # print(f"vertex bytes value: {vertexBytes}")
                x = vertexBytes >> 0 & 0x000007ff
                y = vertexBytes >> 11 & 0x000003ff
                z = vertexBytes >> 21 & 0x000007ff
                vector = [x,y,z]
                # print(f"Vertex vector after bit shift: {vector}")

                lightnormal_start_index = (40 + 6 * header.num_xyz) * current_frame_number + 44 + v * 6
                lightnormal_end_index = (40 + 6 * header.num_xyz) * current_frame_number + 46 + v * 6
                normal = struct.unpack("<H", frames_bytes[lightnormal_start_index : lightnormal_end_index])[0]

            vertex = vertex_t(vector, normal)
            verts.append(vertex)
            

        name = name.rstrip("\x00")
        frame_names.append(name)
        # print(scale, translate, name, verts)
        frame = frame_t(scale, translate, name, verts)
        # print("Frame: ",frame)
        frames.append(frame)
        # print("Frame names ", frame_names)

    # print("Frame Names", frame_names) # write code to count the number of frames in each frame name
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

    # Extra Data
    # After the gl commands lump, there is a short (2 bytes) number for each skin/texture for the model.
    # Each number represents the number of gl commands (not vertices or frames) which the corresponding texture applies to.
    # Simplifying, each gl command is essentially a triangle, but in the order of how open gl would draw them, not the vertex order.
    
    # Prim is abbreviating "primitive," which is another way to refer to a gl command, as it ends up being a primitive shape/polygon.
    # In the case of Quake II/Anachronox models being dealt with here, triangles only.
    header.num_prim = header.num_skins
    header.ofs_prim = header.ofs_glcmds + (4 * header.num_glcmds)

    return header


def load_texture_paths_and_skin_resolutions(skin_names, path):
    # model_path = '\\'.join(path.split('\\')[0:-1])+"\\"
    print(f'Path to file: {path}')
    model_path = str(Path(path).parent)
    print(f'Folder: {model_path}')
    texture_paths = {}
    skin_resolutions = {}
    print("***TEXTURES***")
    for skin_index, skin_name in enumerate(skin_names):
        embedded_texture_name = skin_names[skin_index].rstrip("\x00")
        embedded_texture_name_unextended = os.path.splitext(embedded_texture_name)[0] # remove extension (last one)
        print(f"Embedded Texture Name {embedded_texture_name}")
        """ Look for existing file of given name and supported image format"""
        supported_image_formats = [".png", ".jpg", ".jpeg", ".bmp", ".pcx", ".tga"] # Order doesn't match DP2 image order
        for format in supported_image_formats:
            # Added support for autoloading textures from .md2 directory or a subdirectory with the same name as the embedded texture.
            # and to name Blender mesh the same as .md2 filename if a Display Name is not entered on import screen - Creaper
            texture_path = os.path.join(model_path, embedded_texture_name_unextended) + format
            sub_texture_path = os.path.join(model_path, embedded_texture_name_unextended, embedded_texture_name_unextended) + format

            print(f'Texture Path: {texture_path}')
            print(f'Subtexture Path: {sub_texture_path}')

            if os.path.isfile(texture_path):
                # Get the resolution from the actual image while we're here, as the header only has the first one, which won't cut it for multi-textured models - Holonet
                with PIL.Image.open(texture_path) as img:
                    width, height = img.size
                    skin_resolutions[skin_index] = img.size
                    print(f"Texture found: {texture_path} {skin_resolutions[skin_index]}")
                    texture_paths[skin_index] = texture_path 
                break
            elif os.path.isfile(sub_texture_path):
                    # Get the resolution from the actual image while we're here, as the header only has the first one, which won't cut it for multi-textured models - Holonet
                with PIL.Image.open(sub_texture_path) as img:
                    width, height = img.size
                    skin_resolutions[skin_index] = img.size
                    print(f"Texture found: {sub_texture_path} {skin_resolutions[skin_index]}")
                    texture_path = sub_texture_path
                    texture_paths[skin_index] = texture_path
                break
            else: # if a texture is not located insert a blank texture name into array so the 2nd texture doesn't get moved inplace of the 1st and assign a bum resolution so we don't crash Holonets calculations - Creaper
                skin_resolutions[skin_index] = (64,64)
                texture_paths[skin_index] = ""
        if texture_paths[skin_index] == "":
            print(f"Unable to locate texture for {model_path + '/' + embedded_texture_name}!")
    print("\n")
    print(f"Skin resolution info:\n{skin_resolutions}")
    
    return texture_paths, skin_resolutions



def load_triangle_gl_list(gl_commands, triangles, extra_data):
    """
    This creates a mapping between the gl commands and the triangles.  For models with multiple textures, this is necessary
    because the "extra data" lump indicates how many gl commands the corresponding texture applies to, and these draw calls do not follow the 
    vertex order that we can get from frame data.

    Iterate over the GL COMMANDS
    -> Iterate over the TRIANGLES
    -> -> Iterate over the vertices of the current GL COMMAND
    -> -> If 3 of the vertex indices for that triangle are in the range of vertices for that gl command,
    ...associate that triangle with the appropriate gl command.
    Later, this list is how we will map the triangles to the textures.
    """

    # Dictionary to store which triangles each gl command will be associated with
    triangle_skin_dict = {}
    extra_data_index = 0

    extra_data_total = extra_data[extra_data_index]

    # If there's only 1 skin, there will not be extra data or the need to separate triangles by skin
    if len(extra_data) < 1:
        for triangle_index, triangle in enumerate(triangles):
            triangle_skin_dict[triangle_index] = 0

    else:
        print("Extra data length > 1, handling multiple textures...")
        for command_index, command in enumerate(gl_commands):

            for triangle_index, triangle in enumerate(triangles):
                vertex_match_count = 0
                # Check every vertex index to see if it's in any of the triangle vertex references
                # print(f"Iterating over vertices: {command.vertices}")
                for vert in command.vertices:
                    if triangle.vertexIndices[0] == vert.vertexIndex:
                        # print(f"Vertex {vert} matches triangle {triangle_index}")
                        vertex_match_count += 1
                    if triangle.vertexIndices[1] == vert.vertexIndex:
                        vertex_match_count += 1
                    if triangle.vertexIndices[2] == vert.vertexIndex:
                        vertex_match_count += 1

                if vertex_match_count >= 3:
                    triangle_skin_dict[triangle_index] = extra_data_index
            
            # Each extra data value is a number of gl commands to draw the given texture, NOT an offset
            # So, in order to track when to switch, the extra data total will keep a record of the SUM of these values, so the (gl) command index 
            # can be used to know when we've hit the gl command on which we should proceed to the next texture
            if command_index + 1 == extra_data_total and command_index + 1 != len(gl_commands):
                # print("Incrementing extra data index...")
                # print(f"Gl command index: {command_index}, extra data value: {extra_data[extra_data_index]}")
                extra_data_index += 1
                extra_data_total += extra_data[extra_data_index]
                # print(f"Extra data total: {extra_data_total}")

    # print(triangle_gl_commands)
    print(f"Total of {len(triangle_skin_dict)} triangles associated to a gl command.")

    if len(triangle_skin_dict) < len(triangles):
        print("Issue!  Not all triangles were derived from the gl command vertices!")
        for triangle_index, triangle in enumerate(triangles):
            if triangle_index not in triangle_skin_dict:
                print(f"Triangle {triangle_index} is not in triangle=>skin dictionary!")

    return triangle_skin_dict



def load_texture_coordinates(texture_coordinate_bytes, header, triangles,  skin_resolutions, triangle_gl_dictionary, texture_scale):
    """
    Loads UV (in Quake 2 terms, ST) coordinates
    :param texture_coordinate_bytes:
    :param header:
    :return: list of texture coordinate dataclass objects
    """

    texture_coordinates = list()

    # It seems that lw2md2, used by the ION team, subtracts 1 before creating the S & T coordinates, which scales them incorrectly.  Compensate with this & scale offset below
    coordinate_offset = 0

    for i in range(header.num_st):
        current_coordinate = textureCoordinate_t(*struct.unpack("<hh", texture_coordinate_bytes[4*i:4*i+4]))
        current_coordinate.s += coordinate_offset
        current_coordinate.t += coordinate_offset
        texture_coordinates.append(current_coordinate)

    scale_offset = -2
    texture_skin_dict = {}
    for coord_index, coordinate in enumerate(texture_coordinates):
        for triangle_index, triangle in enumerate(triangles):
            if triangle.textureIndices[0] == coord_index or triangle.textureIndices[1] == coord_index or triangle.textureIndices[2] == coord_index:
                texture_skin_dict[coord_index] = triangle_gl_dictionary[triangle_index] # dictionary returns the skin previously assigned
    
    print(f"texture skin dictionary size: {len(texture_skin_dict)}")

    for coord_index, coord in enumerate(texture_coordinates):
        # -1 here is to account for what seems to be an issue in the creation of the original Anachronox models (as is the coordinate offset above)
        coord.s = coord.s / (((skin_resolutions[texture_skin_dict[coord_index]][0]) / texture_scale) + scale_offset)
        coord.t = coord.t / (((skin_resolutions[texture_skin_dict[coord_index]][1]) / texture_scale) + scale_offset)

    return texture_coordinates


def load_file(path, texture_scale, model_scale):
    """
    Master function returning one dataclass object containing all the MD2 information
    :param path:
    :return:
    """
    with open(path, "rb") as f:  # bsps are binary files
        byte_list = f.read()  # stores all bytes in bytes1 variable (named like that to not interfere with builtin names
    header = load_header(byte_list)
    skin_names = [byte_list[header.ofs_skins + 64 * x:header.ofs_skins + 64 * x + 64].decode("ascii", "ignore") for x in range(header.num_skins)]
    # print(f"Skin Names {skin_names}")

    triangles = load_triangles(byte_list[header.ofs_tris:header.ofs_frames], header)
    frames = load_frames(byte_list[header.ofs_frames:header.ofs_glcmds], header, model_scale)
    texture_paths, skin_resolutions = load_texture_paths_and_skin_resolutions(skin_names, path)

    gl_commands = load_gl_commands(byte_list[header.ofs_glcmds:header.ofs_end_fan])
    
    # This is not the "num_glcmds" from the header.  That number counts each vertex as a separate command.  The following is the number of ACTUAL gl commands
    print(f"Parsed {len(gl_commands)} GL Commands.")

    print(f"\nNumber of primitives: {header.num_prim}")
    print(f"Offset primitives: {header.ofs_prim}\n")

    #tsurf_names = [byte_list[header.ofs_tsurf + 12 * x:header.ofs_tsurf + 12 * x + 8].decode("ascii", "ignore") for x in range(header.num_tsurf)]
    tsurf_dictionary = {}
    for i in range(header.num_tsurf):
        tsurf_start_index = header.ofs_tsurf + (i * 12)
        surface_name = byte_list[tsurf_start_index : tsurf_start_index + 8].decode("ascii", "ignore").rstrip("\x00")
        surface_triange = struct.unpack("<i", byte_list[tsurf_start_index + 8 : tsurf_start_index + 12])[0]
        tsurf_dictionary[surface_name] = surface_triange
    
    print(f"Tagged Surface Names & Triangles:\n{tsurf_dictionary}")

    extra_data = list()
    triangle_skin_dictionary = {}

    primitive_bytes = byte_list[header.ofs_prim : header.ofs_tsurf]
    for i in range(header.num_prim):
        value = struct.unpack("<H", primitive_bytes[0 + (i * 2) : 0 + (i * 2) + 2])[0]
        extra_data.append(value)
        print(f"Extra data value {i+1}: Texture {i+1} applies to {value} gl commands...")

    triangle_skin_dictionary = load_triangle_gl_list(gl_commands, triangles, extra_data)
    
    texture_coordinates = load_texture_coordinates(byte_list[header.ofs_st:header.ofs_tris], header, triangles, skin_resolutions, triangle_skin_dictionary, texture_scale)
    
    for i_frame in range(len(frames)):
        for i_vert in range((header.num_xyz)):
            frames[i_frame].verts[i_vert].v[0] = frames[i_frame].verts[i_vert].v[0] * frames[i_frame].scale.x + frames[i_frame].translate.x
            frames[i_frame].verts[i_vert].v[1] = frames[i_frame].verts[i_vert].v[1] * frames[i_frame].scale.y + frames[i_frame].translate.y
            frames[i_frame].verts[i_vert].v[2] = frames[i_frame].verts[i_vert].v[2] * frames[i_frame].scale.z + frames[i_frame].translate.z

    vertices = list()
    for i, vert in enumerate(frames[0].verts):
        vertices.append(vertex_indexed(i, vert))

    model = md2_object(header, skin_names, triangles, frames, texture_coordinates, gl_commands, tsurf_dictionary, vertices, triangle_skin_dictionary, extra_data, texture_paths)
    return model



def blender_load_md2(md2_path, displayed_name, model_scale, texture_scale, x_rotate, y_rotate, z_rotate, apply_transforms, recalc_normals, use_clean_scene):
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
    ModelVars.x_rotate = x_rotate
    ModelVars.y_rotate = y_rotate
    ModelVars.z_rotate = z_rotate
    ModelVars.apply_transforms = apply_transforms
    ModelVars.recalc_normals = recalc_normals
    ModelVars.md2_path = md2_path


    startProgress("Loading MD2...")

    # If .md2 is missing print an error on screen. This tends to happen if you select an .md2 to load then select another directory path and then select import.
    if not os.path.isfile(md2_path):
        bpy.context.window_manager.popup_menu(missing_file, title="Error", icon='ERROR')
        endProgress()
        return {'FINISHED'} 

    # First lets clean up our scene
    if use_clean_scene:
        for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)

        for block in bpy.data.materials:
            if block.users == 0:
                bpy.data.materials.remove(block)

        for block in bpy.data.textures:
            if block.users == 0:
                bpy.data.textures.remove(block)

        for block in bpy.data.images:
            if block.users == 0:
                bpy.data.images.remove(block)

        for block in bpy.data.actions:
            if block.users == 0:
                bpy.data.actions.remove(block)


    """ Create MD2 dataclass object """
    print(f"md2_path: {md2_path}")
    ModelVars.model_path = '\\'.join(md2_path.split('\\')[0:-1])+"\\"
    print(f"Model Path: {ModelVars.model_path}")
    model_filename = "\\".join(md2_path.split("\\")[-1:])
    print(f"Model Filename: {model_filename}")

    # A dataclass containing all information stored in a .md2 file
    ModelVars.my_object = load_file(md2_path, texture_scale, model_scale)

    """ Create skin path. By default, the one stored inside of the MD2 is used. Some engines like the Digital Paintball 2 one
    check for any image file with that path disregarding the file extension.
    """
    """ get the skin path stored inside of the MD2 """
    # check box must be checked (alternatively it could be checked if the input field was empty or not ...)



    """ Loads required information for mesh generation and UV mapping from the .md2 file"""
    # Gets name to give to the object and mesh in the outliner
    if not displayed_name:
        # Added support for autoloading textures and to name mesh the same as filename if a Display Name is not entered on import screen - Creaper
        # object_name_test = "/".join(object_path.split("/")[-2:]).split(".")[:-1]
        #   
        ModelVars.object_name = os.path.basename(md2_path).split('/')[-1]
        ModelVars.object_name = os.path.splitext(ModelVars.object_name)[0] # remove extension (last one)
        print(f"Blender Outliner Object Name: {ModelVars.object_name}")
        ModelVars.mesh = bpy.data.meshes.new(ModelVars.object_name)  # add the new mesh via filename
        

    else:
        ModelVars.object_name = [displayed_name]
        print(f"Blender Outliner Object Name: {displayed_name}")
        ModelVars.mesh = bpy.data.meshes.new(*ModelVars.object_name)  # add the new mesh, * extracts string from Display Name input list


    # List of vertices [x,y,z] for all frames extracted from the md2 object
    ModelVars.all_verts
    ModelVars.all_verts = [[x.v for x in ModelVars.my_object.frames[y].verts] for y in range(ModelVars.my_object.header.num_frames)]
    # List of vertex indices forming a triangular face
    tris = ([x.vertexIndices for x in ModelVars.my_object.triangles])
    # uv coordinates (in q2 terms st coordinates) for projecting the skin on the model's faces
    # blender flips images upside down when loading so v = 1-t for blender imported images
    uvs_others = ([(x.s, 1-x.t) for x in ModelVars.my_object.texture_coordinates]) 
    # blender uv coordinate system originates at lower left

    """ Lots of code (copy and pasted) that creates a mesh and adds it to the scene collection/outlines """
    ModelVars.obj = bpy.data.objects.new(ModelVars.mesh.name, ModelVars.mesh)
    # col = bpy.data.collections.get("Collection")
    col = bpy.data.collections[0]
    col.objects.link(ModelVars.obj)
    bpy.context.view_layer.objects.active = ModelVars.obj

    # Creates mesh by taking first frame's vertices and connects them via indices in tris
    ModelVars.mesh.from_pydata(ModelVars.all_verts[0], [], tris) 

    """ UV Mapping: Create UV Layer, assign UV coordinates from md2 files for each face to each face's vertices """
    uv_layer=(ModelVars.mesh.uv_layers.new())
    ModelVars.mesh.uv_layers.active = uv_layer
    # add uv coordinates to each polygon (here: triangle since md2 only stores vertices and triangles)
    # note: faces and vertices are stored exactly in the order they were added
    for face_idx, face in enumerate(ModelVars.mesh.polygons):
        for idx, (vert_idx, loop_idx) in enumerate(zip(face.vertices, face.loop_indices)):
            uv_layer.data[loop_idx].uv = uvs_others[ModelVars.my_object.triangles[face_idx].textureIndices[idx]]
                
    """ Create animation for animated models: set keyframe for each vertex in each frame individually """
    # Frame names follow the format: amb_a_001, amb_a_002, etc...
    # A new animation will have a new prefix
    # bpy.context.area.type = "NLA_EDITOR"
    # bpy.context.area.ui_type = "NLA_EDITOR"

    ModelVars.animation_list = list()
    ModelVars.current_anim_name = ""
    frame_count = len(ModelVars.my_object.frames)

    # bpy.ops.wm.create_frames('INVOKE_DEFAULT')

    bpy.ops.amd2_import.macro()

    endProgress()
    return {'FINISHED'} # no idea, seems to be necessary for the UI
        





