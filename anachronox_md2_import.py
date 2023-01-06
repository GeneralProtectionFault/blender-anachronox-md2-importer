bl_info = {
    "name": "Anachronox: Experimental MD2 Importer",
    "author": "Lennart G, Alpaca, Holonet, Creaper",
    "location": "File > Import > Anachronox (.md2)",
    "version": (0, 1, 3),
    "blender": (2, 80, 0),
    "category": "Import-Export"
}

import os
import sys
from dataclasses import dataclass, fields
import struct
from pathlib import Path
from typing import List
import PIL
from PIL import Image, ImagePath
import bpy
from importlib import reload # required when a self-written module is imported that's edited simultaneously
import os  # for checking if skin pathes exist
import math # for applying optional rotate on import

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
        # Get any movement for this frame of animation
        translate = vec3_t(*struct.unpack("<fff", frames_bytes[frame_start + 12 : frame_start + 24]))
        # Animation name
        name = frames_bytes[frame_start + 24 : frame_start + 40].decode("ascii", "ignore")
        

        verts = list()
        # Loop for the number of vertices
        for v in range(header.num_xyz):
            #print(v)
            if header.resolution == 0 or header.resolution == 2:
                # First mess is the vertex (vector)
                vertex_start_index = frame_start + 40 + v * (5 + resolution_bytes)
                vertex_end_index = vertex_start_index + (3 + resolution_bytes)
                # print(f"Vertex index range: {start_index}, {end_index}")
                lightnormal_start_index = frame_start + (43 + resolution_bytes) + v * (5 + resolution_bytes)
                lightnormal_end_index = frame_start + (45 + resolution_bytes) + v * (5 + resolution_bytes)

                # struct.unpack returns tuple--in this case, 3 bytes, which are coordinates for the vertex
                verts.append(vertex_t(list(struct.unpack(unpack_format, frames_bytes[vertex_start_index : vertex_end_index])), # list() only for matching expected type 
                     *struct.unpack("<H", frames_bytes[lightnormal_start_index : lightnormal_end_index])))  

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
    # For each skin, there is a short betwen the end of the glcommands - beginning of the tagged surfaces (Meaning currently unclear but probably opengl-related)
    # This does not appear to be in the header info, but can be derived form it.  These values will be placed here, and could be added to the md2_object if necessary
    
    # This also means the offset primitives is the same as the ofs_end_fan value (should act as a check)
    header.num_prim = header.num_skins
    header.ofs_prim = header.ofs_glcmds + (4 * header.num_glcmds)

    return header


def load_texture_paths_and_skin_resolutions(skin_names, path):
    model_path = '\\'.join(path.split('\\')[0:-1])+"\\"
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
            texture_path = model_path + embedded_texture_name_unextended + format
            sub_texture_path = model_path + embedded_texture_name_unextended + "\\" + embedded_texture_name_unextended + format
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
            else: # if a texture us not located insert a blank texture name into array so the 2nd texture doesn't get moved inplace of the 1st and assign a bum resolution so we don't crash Holonets calculations - Creaper
                skin_resolutions[skin_index] = (64,64)
                texture_paths[skin_index] = ""
        if texture_paths[skin_index] == "":
            print(f"Unable to locate texture for {model_path + embedded_texture_name}!")
    print("\n")
    print(f"Skin resolution info:\n{skin_resolutions}")
    
    return texture_paths, skin_resolutions



def load_triangle_gl_list(gl_commands, triangles, extra_data):
    # Iterate over the GL COMMANDS
    # -> Iterate over the TRIANGLES
    # -> -> Iterate over the vertices of the current GL COMMAND
    # -> -> If at 3 of the vertex indices for that triangle are in the range of vertices for that gl command,
    # ...associate that triangle with one skin or the other, depending on if the gl command is in the range specified
    
    # Dictionary to store which triangles each gl command will be associated with
    triangle_skin_dict = {}

    # If there's only 1 skin, there will not be extra data or the need to separate triangles by skin
    if len(extra_data) < 1:
        for triangle_index, triangle in enumerate(triangles):
            triangle_skin_dict[triangle_index] = 0

    else:
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
                    # Check if the current gl command index is in the ofs_prim => num_prim range, and assign the different skin accordingly
                    # if my_object.extra_data[0][1] <= command_index <= my_object.extra_data[0][1] + my_object.extra_data[0][0]:
                    if extra_data[0][0] <= command_index <= extra_data[0][0] + extra_data[0][1]:
                        # print(f"gl command {command_index} within range, assigning triangle {triangle_index} to skin 0")
                        triangle_skin_dict[triangle_index] = 1
                    else:
                        # print(f"gl command {command_index} not within range, assigning triangle {triangle_index} to skin 1")
                        triangle_skin_dict[triangle_index] = 0

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


def load_file(path, texture_scale):
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
    frames = load_frames(byte_list[header.ofs_frames:header.ofs_glcmds], header)
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
    # If there's only 1 skin, this stuff will not exist and we'll have ourselves a crash, so only get this conditionally
    #if header.num_skins > 1:
    primitive_bytes = byte_list[header.ofs_prim : header.ofs_tsurf]
    for i in range(header.num_prim - 1):
        values = struct.unpack("<HH", primitive_bytes[(i * 2) : (i * 2) + 4])
        extra_data.append(values)
        print(f"Primitive value {i}: {values}")

    triangle_skin_dictionary = load_triangle_gl_list(gl_commands, triangles, extra_data)
    
    texture_coordinates = load_texture_coordinates(byte_list[header.ofs_st:header.ofs_tris], header, triangles, skin_resolutions, triangle_skin_dictionary, texture_scale)
    
    for i_frame in range(len(frames)):
        for i_vert in range((header.num_xyz)):
            frames[i_frame].verts[i_vert].v[0] = frames[i_frame].verts[i_vert].v[0] * frames[i_frame].scale.x + frames[i_frame].translate.x
            frames[i_frame].verts[i_vert].v[1] = frames[i_frame].verts[i_vert].v[1] * frames[i_frame].scale.y + frames[i_frame].translate.y
            frames[i_frame].verts[i_vert].v[2] = frames[i_frame].verts[i_vert].v[2] * frames[i_frame].scale.z + frames[i_frame].translate.z

    # vertices = [x.v for x in frames[0].verts]
    vertices = list()
    for i, vert in enumerate(frames[0].verts):
        vertices.append(vertex_indexed(i, vert))

    model = md2_object(header, skin_names, triangles, frames, texture_coordinates, gl_commands, tsurf_dictionary, vertices, triangle_skin_dictionary, extra_data, texture_paths)
    return model



def blender_load_md2(md2_path, displayed_name, model_scale, texture_scale, x_rotate, y_rotate, z_rotate, apply_transforms, recalc_normals, use_clean_scene):

    # If .md2 is missing print an error on screen. This tends to happen if you select an .md2 to load then select another directory path and then select import.
    if not os.path.isfile(md2_path):
        bpy.context.window_manager.popup_menu(missing_file, title="Error", icon='ERROR')
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
    print(f"md2_path: {md2_path}")
    model_path = '\\'.join(md2_path.split('\\')[0:-1])+"\\"
    print(f"Model Path: {model_path}")
    model_filename = "\\".join(md2_path.split("\\")[-1:])
    print(f"Model Filename: {model_filename}")

    # A dataclass containing all information stored in a .md2 file
    my_object = load_file(md2_path, texture_scale)

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
        object_name = os.path.basename(md2_path).split('/')[-1]
        object_name = os.path.splitext(object_name)[0] # remove extension (last one)
        print(f"Blender Outliner Object Name: {object_name}")
        mesh = bpy.data.meshes.new(object_name)  # add the new mesh via filename
        

    else:
        object_name = [displayed_name]
        print(f"Blender Outliner Object Name: {displayed_name}")
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
    for frame_index in range(my_object.header.num_frames):
        for idx,v in enumerate(obj.data.vertices):
            obj.data.vertices[idx].co = all_verts[frame_index][idx]
            v.keyframe_insert('co', frame=frame_index*2)  # parameter index=2 restricts keyframe to dimension

    # insert first keyframe after last one to yield cyclic animation
    # for idx,v in enumerate(obj.data.vertices):
    # 	obj.data.vertices[idx].co = all_verts[0][idx]
    # 	v.keyframe_insert('co', frame=60)
    

    # New method to assign materials per skin/texture (we only have the single triangle reference for tagged surfaces, and they are not 1-to-1 with skins)
    texpath=0
    print("***MATERIALS***")
    for skin_index, skin in enumerate(my_object.skin_names):
        
        material_name = ("M_" + my_object.skin_names[skin_index].rstrip("\x00"))
        print(f"Blender Material name: {material_name}")
        
        mat = bpy.data.materials.new(name=material_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs['Specular'].default_value = 0
        texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')

        # Give an error and assign a purple color if all textures are missing
        if(my_object.texture_paths == []):
            # Give and error and assign a purple color if one texture is missing
            print(f"Cannot find textures for {md2_path}!")
            print(f"Check {model_path} for .mda material texture file.")
            bsdf.inputs['Base Color'].default_value = (1,0,.5,1)

        if(texpath < len(my_object.texture_paths)):
            if(my_object.texture_paths[skin_index] == ''):
                print(f"Material Texture: MISSING!")
            else:
                print(f"Material Texture: {my_object.texture_paths[skin_index]}")
            if my_object.texture_paths[skin_index] != '':
                texImage.image = bpy.data.images.load(my_object.texture_paths[skin_index])
                mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
                # again copy and paste
            else:
                print(f"Cannot find texture {my_object.texture_paths[triangle_index]}!")
                print(f"Check {model_path} for .mda material texture file.")
                bsdf.inputs['Base Color'].default_value = (1,0,.5,1)

        #bsdf.inputs['Base Color'].default_value = (1,0,.5,1)

        texpath += 1
        mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
        obj.data.materials.append(mat)



    # If there are multiple textures, we need to reassign the triangles 
    if len(my_object.skin_names) > 1:
        bpy.context.tool_settings.mesh_select_mode = [False, False, True]
        
        for material_index, material in enumerate(obj.data.materials):
            bpy.context.object.active_material_index = material_index

            # print(f"Current material index: {bpy.context.object.active_material_index}")
            bpy.ops.object.mode_set(mode = 'EDIT')
            bpy.ops.mesh.select_all(action = 'DESELECT')
            
            skin_triangle_list = list()
            # triangle is key, skin index is the value
            for tri in my_object.triangle_skin_dict:
                if my_object.triangle_skin_dict[tri] == material_index:
                    # print(f"Appending triangle {tri} to list for skin {material_index}")
                    skin_triangle_list.append(tri)


            bpy.ops.object.mode_set(mode = 'OBJECT')
            for face_idx, face in enumerate(mesh.polygons):
                # mesh.polygons[face_idx].select = True
                if face_idx in (skin_triangle_list):
                    face.material_index = bpy.context.object.active_material_index

            # TEST ------------------------------------
            selected_count = 0
            for face_idx, face in enumerate(mesh.polygons):
                if mesh.polygons[face_idx].select == True:
                    selected_count += 1
            
            print(f"Selected face count: {selected_count}")
            # -----------------------------------------

            # bpy.ops.object.mode_set(mode = 'EDIT')
            # bpy.ops.object.material_slot_assign()


    # Apply new scale set on import screen
    print("Seting to object mode...")
    bpy.ops.object.mode_set(mode = 'OBJECT')
    print("Setting origin to geometry...")
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
    # bpy.data.objects[obj_name].scale = (model_scale, model_scale, model_scale)
    obj.scale = (model_scale, model_scale, model_scale)
    print(f"New model scale: { model_scale}")
    print("New model scale applied...")
    bpy.context.active_object.rotation_euler[0] = math.radians(x_rotate) # rotate on import axis=(1=X 2=Y, 3=Z) degrees=(amount)
    bpy.context.active_object.rotation_euler[1] = math.radians(y_rotate) # rotate on import axis=(1=X 2=Y, 3=Z) degrees=(amount)
    bpy.context.active_object.rotation_euler[2] = math.radians(z_rotate) # rotate on import axis=(1=X 2=Y, 3=Z) degrees=(amount)
    print("Object rotated per selected parameters...")

    # Apply Transforms if option selected on import screen
    if(apply_transforms):
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
    if(recalc_normals):
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
   
    # Added a bunch of nifty import options so you do not have to do the same tasks 1000+ times when converting models to another engine. -  Creaper

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
    model_scale: bpy.props.FloatProperty(name="New Model Scale",
                                        description="Desired scale for the model.\nGood for rescaling the model to fit other scale systems. I.E. .0254 is the scale for Unreal Engine.",
                                        default=.0254)
    # Add a texture scale value.  Use this to properly calculate the UV/ST data if someone wants to use an upscaled texture
    texture_scale: bpy.props.FloatProperty(name="Texture Scale",
                                        description="Change to use upscaled textures.\nI.E. If providing 4x textures, set value to 4.",
                                        default=1)          

    # Added support to rotate the model to the desired scale input at import screen
    x_rotate: bpy.props.FloatProperty(name="X-axis Rotate",
                                        description="Rotation adjusment on X-axis for the model.\nGood for if you need models rotated and don't want to manually do it for each model upon import.",
                                        soft_max=360,
                                        soft_min=-360)
    y_rotate: bpy.props.FloatProperty(name="Y-axis Rotate",
                                        description="Rotation adjusment on Y-axis for the model.\nGood for if you need models rotated and don't want to manually do it for each model upon import.",
                                        soft_max=360,
                                        soft_min=-360)
    z_rotate: bpy.props.FloatProperty(name="Z-axis Rotate",
                                        description="Rotation adjusment on Z-axis for the model.\nGood for if you need models rotated and don't want to manually do it for each model upon import.",
                                        soft_max=360,
                                        soft_min=-360)
 
    # Added option to apply all of the above transforms. Some issue is making it not work quite right yet
    apply_transforms: BoolProperty(name="Apply transforms",
                                        description="Applies the previous transforms.\nIf you need the scale and rotation transforms applied upon import select this.",
                                        default=False)

    # Added option to flip normals as they seem to be inside upon import
    recalc_normals: BoolProperty(name="Recalc. Normals",
                                        description="Recalculate normals-outside.\nYou typically want this set as Anachronox normals are opposite what they are in Blender.",
                                        default=False)

    # Added option to clean the Blender scene of unused items do you don't end up with a bunch of stuff named .### and have to manually rename them
    use_clean_scene: BoolProperty(name="Clean Scene",
                                        description="Clean the Blender scene of any unused data blocks including unused Materials, Textures and Names.\nYou typically want this set.",
                                        default=True)

    
    def execute(self, context):
        try:
            return blender_load_md2(self.filepath, self.displayed_name, self.model_scale, self.texture_scale, self.x_rotate, self.y_rotate, self.z_rotate, self.apply_transforms, self.recalc_normals, self.use_clean_scene)
        except Exception as argument:
            self.report({'ERROR'}, argument)

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

def missing_file(self, context):
    self.layout.label(text="Model file does not exist in currently selected directory! Perhaps you didn't select the correct .md2 file?")

if __name__ == "__main__":
    
    register()

    # test call
    bpy.ops.import_md2.some_data('INVOKE_DEFAULT')
