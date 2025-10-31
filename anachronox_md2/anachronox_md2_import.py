import bpy
import os
import stat
import sys
import subprocess
from dataclasses import dataclass, fields
import struct
from pathlib import Path
from typing import List
import platform
import re
from difflib import SequenceMatcher
from collections import defaultdict

from .utils import *

from importlib import reload # required when a self-written module is imported that's edited simultaneously
import math # for applying optional rotate on import
# import mathutils
# from mathutils import Vector

SUPPORTED_IMAGE_FORMATS = [".png", ".jpg", ".jpeg", ".bmp", ".pcx", ".tga"]     # Texture file formats


# path to python.exe
if platform.system() == "Linux":
    # Depending on the environment, the binary might be "python" or "python3.11", etc...
    # Stupid...but need to "find" the python binary to avoid a crash...
    python_bin_folder = os.path.join(sys.prefix, 'bin')

    # Search for binary files
    executable = stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
    for filename in os.listdir(python_bin_folder):
        full_python_path = os.path.join(python_bin_folder, filename)
        if os.path.isfile(full_python_path):
            st = os.stat(full_python_path)
            mode = st.st_mode
            # If file is an executable and contains the text "python"
            if mode & executable and 'python' in filename:
                # print(filename,oct(mode))
                break

    python_exe = full_python_path
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



def load_gl_commands(gl_command_bytes):
    """
    Loads gl_commands which are a list of GL_TRIANGLE_STRIP and GL_TRIANGLE_FAN calls that reduce fps
    Code differs much from original loading code in C
    :param gl_command_bytes: bytes belonging to gl_commands lump from md2 file
    :return: list of dataclasses storing gl commands
    """
    offset = 0
    ModelVars.gl_commands = list()
    while True:  # ends when mode is 0
        # This "mode" is just an integer.  If positive, it's GL_TRIANGLE_STRIP.  If negative, it's GL_TRIANGLE_FAN.  If 0, end of the lump
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
        ModelVars.gl_commands.append(glCommand_t(mode, gl_vertices))
    return ModelVars.gl_commands


def load_triangles(triangle_bytes):
    """
    Creates basic list of triangle dataclasses which contain indices to vertices
    :param triangle_bytes: bytes from md2 file belonging to triangles lump
    :param header: dataclass containing header information
    :return: list of triangles
    """
    ModelVars.triangles = list()
    for i in range(ModelVars.header.num_tris):
        triangle = triangle_t(list(struct.unpack("<hhh", triangle_bytes[12*i:12*i+6])), list(struct.unpack("<hhh", triangle_bytes[12*i+6:12*i+12])))
        ModelVars.triangles.append(triangle)



def load_frames(frames_bytes):
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
    ModelVars.frames = list()
    frame_names = list()
    if ModelVars.header.resolution == 0: # vertex information in 3 bytes (8 bit vertices)
        unpack_format = "<BBB"
        resolution_bytes = 0
    elif ModelVars.header.resolution == 1: # vertex information in 4 bytes (11 bit X, 10 bit Y, 11 bit Z)
        unpack_format = "<BBB"
        resolution_bytes = 1
    elif ModelVars.header.resolution == 2: # vertex information in 6 bytes (16 bit vertices)
        unpack_format = "<HHH"
        resolution_bytes = 3

    for current_frame_number in range(ModelVars.header.num_frames):

        frame_start = (40 + (5 + resolution_bytes) * ModelVars.header.num_xyz) * current_frame_number

        # Get any scaling for this frame of animation
        scale = vec3_t(*struct.unpack("<fff", frames_bytes[frame_start : frame_start + 12]))


        # print(scale)
        # Get any movement for this frame of animation
        translate = vec3_t(*struct.unpack("<fff", frames_bytes[frame_start + 12 : frame_start + 24]))

        # print(f"Scale Before: {scale}")
        # print(f"Translate Before: {translate}")

        ##### This applies the scaling set in the UI when importing ##############################################
        scale = vec3_t(x = scale.x * ImportOptions.model_scale, y = scale.y * ImportOptions.model_scale, z = scale.z * ImportOptions.model_scale)
        translate = vec3_t(translate.x * ImportOptions.model_scale, translate.y * ImportOptions.model_scale, translate.z * ImportOptions.model_scale)
        ##########################################################################################################

        # print(f"Scale After: {scale}")
        # print(f"Translate After: {translate}")

        # Animation name
        name = frames_bytes[frame_start + 24 : frame_start + 40].decode("ascii", "ignore")

        verts = list()
        # print(f"Vertex Resolution: {header.resolution}")

        # Loop for the number of vertices
        for v in range(ModelVars.header.num_xyz):
            if ModelVars.header.resolution == 0 or ModelVars.header.resolution == 2:
                # First mess is the vertex (vector)
                vertex_start_index = frame_start + 40 + v * (5 + resolution_bytes)
                vertex_end_index = vertex_start_index + (3 + resolution_bytes)
                # print(f"Vertex index range: {start_index}, {end_index}")
                lightnormal_start_index = frame_start + (43 + resolution_bytes) + v * (5 + resolution_bytes)
                lightnormal_end_index = frame_start + (45 + resolution_bytes) + v * (5 + resolution_bytes)

                # struct.unpack returns tuple--in this case, 3 bytes, which are coordinates for the vertex
                vector = list(struct.unpack(unpack_format, frames_bytes[vertex_start_index : vertex_end_index])) # list() only for matching expected type
                normal = struct.unpack("<H", frames_bytes[lightnormal_start_index : lightnormal_end_index])


            elif ModelVars.header.resolution == 1:
                vertex_start_index = (40 + 6 * ModelVars.header.num_xyz) * current_frame_number + 40 + v * 6
                vertexBytes = int.from_bytes(frames_bytes[vertex_start_index : vertex_start_index + 4], sys.byteorder)
                # print(f"vertex bytes value: {vertexBytes}")
                x = vertexBytes >> 0 & 0x000007ff
                y = vertexBytes >> 11 & 0x000003ff
                z = vertexBytes >> 21 & 0x000007ff
                # print(f"Vector bytes before bitshift: {vertexBytes}")
                vector = [x,y,z]
                # print(f"Vertex vector after bit shift: {vector}")

                lightnormal_start_index = (40 + 6 * ModelVars.header.num_xyz) * current_frame_number + 44 + v * 6
                lightnormal_end_index = (40 + 6 * ModelVars.header.num_xyz) * current_frame_number + 46 + v * 6
                normal = struct.unpack("<H", frames_bytes[lightnormal_start_index : lightnormal_end_index])[0]

            # print(f"Vector: {vector}")
            vertex = vertex_t(vector, normal)
            verts.append(vertex)


        name = name.rstrip("\x00")
        frame_names.append(name)
        # print(scale, translate, name, verts)
        frame = frame_t(scale, translate, name, verts)
        # print("Frame: ",frame)
        ModelVars.frames.append(frame)
        # print("Frame names ", frame_names)

    # print("Frame Names", frame_names) # write code to count the number of frames in each frame name
    return ModelVars.frames


def load_header(file_bytes):
    """
    Creates header dataclass object
    :param file_bytes: bytes from md2 file belonging to header
    :return: header dataclass object
    """
    # print(file_bytes[:4].decode("ascii", "ignore"))
    arguments = struct.unpack("<ihhiiiiiiiiiiiiiiiiifffii", file_bytes[:96])
    ModelVars.header = md2_t(*arguments)
    # Verify MD2
    if not ModelVars.header.ident == 844121161 or not ModelVars.header.version == 15:
        print(f"Error: File type is not MD2. Ident or version not matching")
        print(f'Ident: {file_bytes[:4].decode("ascii", "ignore")} should be "IDP2"')
        print(f"Version: {ModelVars.header.version} should be 15")


    print("--------------- HEADER VALUES -------------------")
    for field in fields(ModelVars.header):
        print(f"{field.name} - ", getattr(ModelVars.header, field.name))

    print("--------------------------------------------------")

    # Extra Data
    # After the gl commands lump, there is a short (2 bytes) number for each skin/texture for the model.
    # Each number represents the number of gl commands (not vertices or frames) which the corresponding texture applies to.
    # Simplifying, each gl command is essentially a triangle, but in the order of how open gl would draw them, not the vertex order.

    # Prim is abbreviating "primitive," which is another way to refer to a gl command, as it ends up being a primitive shape/polygon.
    # In the case of Quake II/Anachronox models being dealt with here, triangles only.
    ModelVars.header.num_prim = ModelVars.header.num_skins
    ModelVars.header.ofs_prim = ModelVars.header.ofs_glcmds + (4 * ModelVars.header.num_glcmds)



def _try_load_texture_from_base(base_path_for_texture_stem: Path, supported_formats: list[str]):
    """
    Tries to load a texture by appending supported formats to a base path (which includes the stem).
    Args:
        base_path_for_texture_stem (Path): The base path and name of the texture 
                                           (e.g., Path("textures/folder/mytexture_stem")).
        supported_formats (list of str): List of supported image extensions.
    Returns:
        tuple (Path, tuple) or (None, None): Found texture path and (width, height), or None if not found.
    """
    for fmt in supported_formats:
        candidate_path = base_path_for_texture_stem.with_suffix(fmt)

        if candidate_path.is_file():
            try:
                with PIL.Image.open(candidate_path) as img:
                    # Resolution from actual image
                    print(f"✅ Texture found: {candidate_path} {img.size}")
                    return candidate_path, img.size
            except PIL.UnidentifiedImageError:
                print(f"⛔ Warning: File {candidate_path} found but is not a recognized image or is corrupted.")
            except Exception as e: # pylint: disable=broad-except
                print(f"⛔ Warning: Error opening image {candidate_path}: {e}")
    return None, None


def extract_file_value(file_path):
    """
    Extracts the value of the `file` field for the `#0 > !bitmap > file` construct from the text file.

    :param file_path: Path to the text file
    :return: Value of the `file` field, or None if not found
    """
    with open(file_path, 'r') as file:
        content = file.read()

    # Regular expression to match the specific construct
    pattern = r"#\s*0\s*!bitmap\s*file\s*=\s*(.+)"

    match = re.search(pattern, content)
    if match:
        return match.group(1).strip()
    return None


def parse_material_file(file_path):
    """
    Reads a text file, locates all "map" element values inside the
    structure profile -> skin -> pass, and groups them by "profile" value.
    """

    # Use defaultdict to automatically create list for new profiles
    results = defaultdict(list)

    current_profile_name = None

    # Stack to keep track of block types and the brace level they opened at.
    # Each element: (block_type_str, brace_level_opened_at)
    # block_type_str can be 'profile', 'skin', 'pass', or 'anonymous' (for unrecognized blocks)
    block_context_stack = [] 
    current_brace_level = 0

    # These variables help associate a keyword (like 'profile', 'skin', 'pass')
    # with the next '{' that opens its block.
    pending_keyword = None
    pending_profile_name_candidate = None # Specifically for 'profile' keyword

    # Regex to find 'map "some/path/here"' and capture "some/path/here"
    # It allows for optional whitespace after "map"
    map_regex = re.compile(r'map\s+"(.*?)"')

    try:
        with open(file_path, 'r') as f:
            spike = False
            for line_num, raw_line in enumerate(f, 1):
                line = raw_line.strip()

                if not line or line.startswith("//"): # Skip empty lines and C-style comments
                    continue

                # 1. Check for keywords that define blocks we care about.
                # These set a "pending" state, waiting for a '{'.
                if line.startswith("profile"):
                    parts = line.split()
                    if len(parts) > 1:
                        pending_keyword = "profile"
                        pending_profile_name_candidate = parts[1]
                    else: # Malformed profile line
                        pending_keyword = "profile" 
                        pending_profile_name_candidate = "EMPTY"
                elif line.startswith("skin"):
                    # Only consider 'skin' if we are inside a profile context,
                    # or if it's a pending keyword to be validated when '{' appears.
                    pending_keyword = "skin"
                elif line.startswith("pass"):
                    # Similar logic for 'pass', needs to be in a 'skin' context.
                    pending_keyword = "pass"

                # 2. Check for "map" lines.
                # This must be within the specific hierarchy: profile -> skin -> pass.
                # The regex match is done here, independent of whether this line also opens/closes a block.
                map_match = map_regex.match(line) # Use match() as 'map' should be at the start
                if map_match:
                    if current_profile_name and len(block_context_stack) >= 2:
                        # Check context: top of stack should be 'pass', one below should be 'skin'.
                        # The 'profile' context is confirmed by 'current_profile_name' being set.
                        if block_context_stack[-1][0] == 'pass' and \
                           block_context_stack[-2][0] == 'skin':
                            map_value = map_match.group(1)

                            # Some MDAs use forward slashes, some use back slashes :-| :-| :-|...
                            map_value = map_value.replace('\\','/')

                            # print(f"MAP VALUE ------------- {map_value}")
                            results[current_profile_name].append(map_value)
                            spike = True

                # 3. Handle block opening '{'.
                # This can be on its own line or at the end of a keyword line (e.g., "skin {").
                if line.endswith("{"):
                    block_opened_this_line = True # Flag that a block was opened
                    # Use the pending_keyword (if any) to identify the block type.
                    if pending_keyword == "profile":
                        current_profile_name = pending_profile_name_candidate
                        # Ensure the list for this profile exists
                        if current_profile_name not in results:
                             results[current_profile_name] = []
                        block_context_stack.append(("profile", current_brace_level))
                    elif pending_keyword == "skin":
                        # Skin must be within an active profile block or its descendant
                        if current_profile_name: 
                            block_context_stack.append(("skin", current_brace_level))
                            spike = False
                        else: # Skin outside profile, treat as anonymous or ignore
                            block_context_stack.append(("anonymous", current_brace_level))
                    elif pending_keyword == "pass":
                        # Pass must be within an active skin block
                        if current_profile_name and block_context_stack and \
                           block_context_stack[-1][0] == 'skin' and not spike:
                            block_context_stack.append(("pass", current_brace_level))
                        else: # Pass outside skin, treat as anonymous or ignore
                            block_context_stack.append(("anonymous", current_brace_level))
                    elif pending_keyword is None and line == "{": 
                        # An opening brace not preceded by a tracked keyword: anonymous block
                        block_context_stack.append(("anonymous", current_brace_level))
                    # else: A keyword was pending but the line was e.g. "keyword {" and keyword wasn't tracked
                    #       This case is covered by the above, or it's an anonymous block if pending_keyword was None.

                    current_brace_level += 1
                    pending_keyword = None # The '{' has consumed the pending keyword
                    pending_profile_name_candidate = None

                # 4. Handle block closing '}'.
                elif line == "}":
                    if current_brace_level > 0: # Ensure we don't go negative
                        current_brace_level -= 1
                    else:
                        # This indicates a syntax error (more '}' than '{')
                        # print(f"Warning: Unmatched '}}' at line {line_num_val} in {file_path}")
                        pass # Or raise error

                    if block_context_stack and block_context_stack[-1][1] == current_brace_level:
                        # This '}' closes the block at the top of the stack
                        closed_block_type, _ = block_context_stack.pop()
                        if closed_block_type == "profile":
                            current_profile_name = None # Exited current profile's scope
                    # else: Mismatched brace or closing an untracked anonymous block.

                    pending_keyword = None # A '}' also clears any pending keyword expectation.
                    pending_profile_name_candidate = None

                # 5. If a line is not a keyword starter we track, and not a brace,
                #    and didn't open a block this line, it might be an inner statement
                #    like 'sort blend'. Such lines should not clear a pending_keyword
                #    if it was set by a *previous* line and waiting for a '{' on a *future* line.
                #    The logic already handles this: pending_keyword is only cleared by
                #    '{', '}', or being overwritten by another profile/skin/pass keyword.
                #
                #    If a line *was* a keyword starter (e.g. "skin") but did *not* end with "{",
                #    pending_keyword remains set, waiting for a "{" on a subsequent line.
                #    If the line *was not* a keyword starter, and did not end with "{",
                #    pending_keyword from a previous line (if any) persists.

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred during parsing: {e} (line {line_num})")
        return None

    return dict(results) # Convert defaultdict to dict for cleaner output if preferred


def truncate_after(lst, marker):
    if marker in lst:
        return lst[:lst.index(marker)]
    return lst[:]  # unchanged if marker not found


def merge_paths(base_path, relative_path):
    """
    Merges a base path with a relative path, replacing the portion
    of the base path indicated by the relative path.

    Args:
        base_path (str): The original base path (e.g., 'c:\\a\\b\\c\\d\\e').
        relative_path (str): The relative path to merge (e.g., '\\c\\d\\f').

    Returns:
        str: The resulting merged path (e.g., 'c:\\a\\b\\c\\d\\f').
    """

    # Normalize paths to handle different formats and separators
    base_parts = os.path.normpath(base_path).split(os.sep)
    relative_parts = os.path.normpath(relative_path).lstrip(os.sep).split(os.sep) 

    # We need this to be the "models" directory, as referred to in the MDA files, etc...
    # Problem is, people could have extraced these anywhere, so the "models" folder needs to be whatever the main folder MODELS.DAT content was extracted to
    # Ergo, we'll try to accomodate all the folder possibilities here... 

    # All 1st-level folders
    folders_known_in_main_extracted_location = ['background','boots','cine','cube2','cursor','democratus','fatima','fx','grumpos','grumpos2x', \
    'interface','intro','monsters','mystech','newface','npcs','objects','paco','pal','proc','rho','stiletto']

    # Get the first matching item
    match = next((item for item in folders_known_in_main_extracted_location if item in base_parts), None)
    if match:
        base_parts = truncate_after(base_parts, match)

    

    if "models" in relative_parts:
        relative_parts.remove("models") # "models" appears to be a parent folder that we should replace w/ wherever files are extracted to

    print(f"\nBASE PARTS: {base_parts}")
    print(f"RELATIVE PARTS: {relative_parts}\n")

    try:
        # Find the index in the base path where the relative path starts
        if any(item in base_parts for item in relative_parts):
            start_index = base_parts.index(relative_parts[0])

            # Replace the corresponding parts of the base path with the relative path
            merged_path = os.path.join(*base_parts[:start_index], *relative_parts)
        else:
            merged_path = os.path.join(*base_parts, *relative_parts)

        # Detect whether the original base path was absolute.
        #  * On Windows:  os.path.splitdrive returns ('C:', '\\path\\to\\file')
        #  * On POSIX:    splitdrive returns ('', '/path/to/file')

        if os.path.isabs(base_path):
            drive, _ = os.path.splitdrive(base_path)     # e.g. 'C:'  or  ''
            root = drive if drive else os.sep             # 'C:' on Windows, '/' on Linux

            # If the join removed the root, re‑add it.
            if not merged_path.startswith(root):
                # Strip *any* leading separator that the join might have left
                merged_path = os.path.join(root, merged_path.lstrip(os.sep))

        return merged_path
    except ValueError as e:
        print(f"ERROR MERGING PATHS: {e}")
        # If the relative path doesn't match the base path, return as is
        return os.path.join(base_path, relative_path)


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
    extra_data_index = 0
    extra_data_total = ModelVars.extra_data[extra_data_index]

    # If there's only 1 skin, there will not be extra data or the need to separate triangles by skin
    if len(ModelVars.extra_data) < 1:
        for triangle_index, triangle in enumerate(ModelVars.triangles):
            ModelVars.triangle_skin_dict[triangle_index] = 0

    else:
        print("Extra data length > 1, handling multiple textures...")
        for command_index, command in enumerate(ModelVars.gl_commands):

            for triangle_index, triangle in enumerate(ModelVars.triangles):
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
                    ModelVars.triangle_skin_dict[triangle_index] = extra_data_index

            # Each extra data value is a number of gl commands to draw the given texture, NOT an offset
            # So, in order to track when to switch, the extra data total will keep a record of the SUM of these values, so the (gl) command index 
            # can be used to know when we've hit the gl command on which we should proceed to the next texture
            if command_index + 1 == extra_data_total and command_index + 1 != len(ModelVars.gl_commands):
                # print("Incrementing extra data index...")
                # print(f"Gl command index: {command_index}, extra data value: {ModelVars.extra_data[extra_data_index]}")
                extra_data_index += 1
                extra_data_total += ModelVars.extra_data[extra_data_index]
                # print(f"Extra data total: {extra_data_total}")

    # print(triangle_gl_commands)
    print(f"Total of {len(ModelVars.triangle_skin_dict)} triangles associated to a gl command.")

    if len(ModelVars.triangle_skin_dict) < len(ModelVars.triangles):
        print("Issue!  Not all triangles were derived from the gl command vertices!")
        for triangle_index, triangle in enumerate(ModelVars.triangles):
            if triangle_index not in ModelVars.triangle_skin_dict:
                print(f"Triangle {triangle_index} is not in triangle=>skin dictionary!")

    return ModelVars.triangle_skin_dict


def load_texture_coordinates(texture_coordinate_bytes, header, triangles, skin_resolutions, triangle_gl_dictionary, texture_scale):
    """
    Loads UV (in Quake 2 terms, ST) coordinates
    :param texture_coordinate_bytes:
    :param header:
    :return: list of texture coordinate dataclass objects
    """

    ModelVars.texture_coordinates = list()

    # It seems that lw2md2, used by the ION team, subtracts 1 before creating the S & T coordinates, which scales them incorrectly.  Compensate with this & scale offset below
    coordinate_offset = 0

    for i in range(ModelVars.header.num_st):
        current_coordinate = textureCoordinate_t(*struct.unpack("<hh", texture_coordinate_bytes[4*i:4*i+4]))
        current_coordinate.s += coordinate_offset
        current_coordinate.t += coordinate_offset
        ModelVars.texture_coordinates.append(current_coordinate)
        # print(f"{i} - {current_coordinate}")

    scale_offset = -2
    texture_skin_dict = {}
    # Iterate over all coordinates
    for coord_index, coordinate in enumerate(ModelVars.texture_coordinates):
        # For every coordinate, check all the triangles...
        for triangle_index, triangle in enumerate(ModelVars.triangles):
            # If any of the 3 verts (by index) of the triangle are this coordinate...
            if triangle.textureIndices[0] == coord_index or triangle.textureIndices[1] == coord_index or triangle.textureIndices[2] == coord_index:
                # Associate this TEXTURE with this coordinate - this is necessary because the UVs need to account for different texture sizes on multiple-texture models
                texture_skin_dict[coord_index] = triangle_gl_dictionary[triangle_index] # dictionary returns the skin previously assigned

    # print(f"texture skin dictionary size: {len(texture_skin_dict)}")

    # Get the s,t (u,v) for each coordinate
    # This uses the above association of coordinate to texture, calculating the UV to be proportional to the width & height of the texture it goes with
    for coord_index, coord in enumerate(ModelVars.texture_coordinates):
        # skin_resolutions returns the x & y size of the texture in question
        # offset here is to account for what seems to be an issue in the creation of the original Anachronox models (as is the coordinate offset above)
        coord.s = coord.s / (((ModelVars.skin_resolutions[texture_skin_dict[coord_index]][0]) / ImportOptions.texture_scale) + scale_offset)
        coord.t = coord.t / (((ModelVars.skin_resolutions[texture_skin_dict[coord_index]][1]) / ImportOptions.texture_scale) + scale_offset)
        # print(f"Post Calculation {coord_index}: S: {coord.s}, T: {coord.t}")


def blender_load_md2():
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
    print("Loading MD2...")

    # If .md2 is missing print an error on screen. This tends to happen if you select an .md2 to load then select another directory path and then select import.
    if not os.path.isfile(ImportOptions.filepath):
        bpy.context.window_manager.popup_menu(missing_file, title="Error", icon='ERROR')
        # endProgress()
        return

    # First lets clean up our scene
    if ImportOptions.use_clean_scene:
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
    print(f"md2_path: {ImportOptions.filepath}")
    ModelVars.model_path = '\\'.join(ImportOptions.filepath.split('\\')[0:-1])+"\\"
    print(f"Model Path: {ModelVars.model_path}")
    ModelVars.model_filename = "\\".join(ImportOptions.filepath.split("\\")[-1:])
    print(f"Model Filename: {ModelVars.model_filename}")


    with open(ImportOptions.filepath, "rb") as f:
        ModelVars.byte_list = f.read()  # stores all bytes in bytes1 variable (named like that to not interfere with builtin names
    load_header(ModelVars.byte_list)
    ModelVars.skin_names = [ModelVars.byte_list[ModelVars.header.ofs_skins + 64 * x:ModelVars.header.ofs_skins + 64 * x + 64].decode("ascii", "ignore") for x in range(ModelVars.header.num_skins)]
    # print(f"Skin Names {ModelVars.skin_names}")

    load_triangles(ModelVars.byte_list[ModelVars.header.ofs_tris:ModelVars.header.ofs_frames])
    ModelVars.frames = load_frames(ModelVars.byte_list[ModelVars.header.ofs_frames:ModelVars.header.ofs_glcmds])

    ModelVars.gl_commands = load_gl_commands(ModelVars.byte_list[ModelVars.header.ofs_glcmds:ModelVars.header.ofs_glcmd_counts])

    # This is not the "num_glcmds" from the header.  That number counts each vertex as a separate command.  The following is the number of ACTUAL gl commands
    print(f"Parsed {len(ModelVars.gl_commands)} GL Commands.")

    print(f"\nNumber of primitives: {ModelVars.header.num_prim}")
    print(f"Offset primitives: {ModelVars.header.ofs_prim}\n")

    ModelVars.tsurf_dictionary = {}
    for i in range(ModelVars.header.num_tsurf):
        tsurf_start_index = ModelVars.header.ofs_tsurf + (i * 12)
        surface_name = ModelVars.byte_list[tsurf_start_index : tsurf_start_index + 8].decode("ascii", "ignore").rstrip("\x00")
        surface_triange = struct.unpack("<i", ModelVars.byte_list[tsurf_start_index + 8 : tsurf_start_index + 12])[0]
        ModelVars.tsurf_dictionary[surface_name] = surface_triange

    print(f"Tagged Surface Names & Triangles:\n{ModelVars.tsurf_dictionary}")

    ModelVars.extra_data = list()
    ModelVars.triangle_skin_dict = {}

    primitive_bytes = ModelVars.byte_list[ModelVars.header.ofs_prim : ModelVars.header.ofs_tsurf]
    for i in range(ModelVars.header.num_prim):
        value = struct.unpack("<H", primitive_bytes[0 + (i * 2) : 0 + (i * 2) + 2])[0]
        ModelVars.extra_data.append(value)
        print(f"Extra data value {i+1}: Texture {i+1} applies to {value} gl commands...")

    ModelVars.triangle_skin_dict = load_triangle_gl_list(ModelVars.gl_commands, ModelVars.triangles, ModelVars.extra_data)


def assign_texture_paths_by_profile(map_key_to_use):
    model_file_path_obj = Path(ImportOptions.filepath)
    initial_model_dir = model_file_path_obj.parent

    for skin_index, skin_name_raw in enumerate(ModelVars.skin_names):
        texture_list_for_key = ModelVars.grouped_maps[map_key_to_use]
        # print(f"TEXTURE LIST FOR KEY: {texture_list_for_key}")

        if isinstance(texture_list_for_key, list) and 0 <= skin_index < len(texture_list_for_key):
            mda_texture_rel_path_str = texture_list_for_key[skin_index]
        else:
            mda_texture_rel_path_str = texture_list_for_key[0]
        if mda_texture_rel_path_str:
            print(f"MDA relative path: {mda_texture_rel_path_str}")
            current_texture_search_base_str = merge_paths(str(initial_model_dir), str(mda_texture_rel_path_str))
            print(f"ℹ️ Path after MDA merge: {current_texture_search_base_str}")

            mda_derived_tex_base = Path(current_texture_search_base_str)
            print(f'ℹ️ Texture Path (from MDA): {mda_derived_tex_base} ({",".join(SUPPORTED_IMAGE_FORMATS)})')
            found_texture_file_path, found_texture_resolution = _try_load_texture_from_base(
                mda_derived_tex_base, SUPPORTED_IMAGE_FORMATS
            )

            if found_texture_file_path:
                ModelVars.texture_paths[skin_index] = str(found_texture_file_path)
                ModelVars.skin_resolutions[skin_index] = found_texture_resolution


def get_texture_paths():
    ModelVars.texture_paths = {}
    ModelVars.skin_resolutions = {}
    ModelVars.grouped_maps = {}
    ModelVars.multiple_profiles = False
    ModelVars.map_key_to_use = None
    ModelVars.mda_path = ""

    print("***TEXTURES***")

    model_file_path_obj = Path(ImportOptions.filepath)
    initial_model_dir = model_file_path_obj.parent
    model_filename_stem = model_file_path_obj.stem

    for skin_index, skin_name_raw in enumerate(ModelVars.skin_names):
        print(f"Getting textures for skin {skin_index} : {skin_name_raw}")
        print(f'ℹ️ Path to file: {model_file_path_obj}')

        # This variable mirrors 'model_path' from original, which changes based on MDA/ATD
        current_texture_search_base_str = str(initial_model_dir)
        print(f'ℹ️ Folder: {current_texture_search_base_str}')

        embedded_texture_name = skin_name_raw.rstrip("\x00")
        # remove extension (last one)
        embedded_texture_name_stem = Path(embedded_texture_name).stem
        print(f"ℹ️ Embedded Texture Name {embedded_texture_name}")

        print('----------------------------------------------------------------------------------------------------------------')
        found_texture_file_path = None
        found_texture_resolution = None

        # Attempt 1: Look for existing file of given name and supported image format
        # Based on embedded texture name in the model's directory or a subdirectory

        # Path in .md2 directory
        # Added support for autoloading textures from .md2 directory
        texture_base_candidate1 = initial_model_dir / embedded_texture_name_stem
        print(f'ℹ️ Texture Path: {texture_base_candidate1} ({",".join(SUPPORTED_IMAGE_FORMATS)})')
        found_texture_file_path, found_texture_resolution = _try_load_texture_from_base(
            texture_base_candidate1, SUPPORTED_IMAGE_FORMATS
        )

        if not found_texture_file_path:
            # Path in subdirectory with the same name as the embedded texture
            # Added support for autoloading textures from a subdirectory with the same name as the embedded texture
            texture_base_candidate2 = initial_model_dir / embedded_texture_name_stem / embedded_texture_name_stem
            print(f'ℹ️ Subtexture Path: {texture_base_candidate2}({",".join(SUPPORTED_IMAGE_FORMATS)})')
            found_texture_file_path, found_texture_resolution = _try_load_texture_from_base(
                texture_base_candidate2, SUPPORTED_IMAGE_FORMATS
            )
        print('----------------------------------------------------------------------------------------------------------------')

        # if a texture is not located insert a blank texture name into array ... and assign a bum resolution
        if not found_texture_file_path:
            ModelVars.skin_resolutions[skin_index] = (64, 64)
            ModelVars.texture_paths[skin_index] = ""
        else:
            ModelVars.texture_paths[skin_index] = str(found_texture_file_path)
            ModelVars.skin_resolutions[skin_index] = found_texture_resolution


        # --- MDA Fallback ---
        if ModelVars.texture_paths[skin_index] == "":
            print("❌ Unable to locate texture, trying to find associated MDA file...")
            character_folder = initial_model_dir.parent     # Hopefully...folder that contains the model, skin, etc...

            # Look for MDA in current folder
            ModelVars.mda_path = initial_model_dir / (model_filename_stem + ".mda")
            if not ModelVars.mda_path.is_file():
                # Look for MDA in parent folder
                ModelVars.mda_path = initial_model_dir.parent / (model_filename_stem + ".mda")

            # MDA is not ALWAYS named the same as the model file X-[...
            # This will check for the MDA matching the FOLDER that corresponds to the character name.
            # Example is rictus, which has rictus.mda, but the model is named kraptonguy.  krap...
            if not ModelVars.mda_path.is_file():
                ModelVars.mda_path = character_folder / Path(character_folder.name + ".mda")

            # Account for naming convention break by taking the closest MDA
            # Example: detta2g_cine.md2 & dettag2_cine.mda (2 & g switched...
            if not ModelVars.mda_path.is_file():
                # Get any files ending in .mda in the main character's folder
                mda_files = [str(f) for f in character_folder.glob("**/*.mda")]
                print(f"Checking any mda files: {mda_files}\nfor similarity to model file: {model_filename_stem}")
                best = None
                best_ratio = -1.0
                for f in mda_files:
                    ratio = SequenceMatcher(None, model_filename_stem, Path(f).name).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best = f
                ModelVars.mda_path = Path(best)

                print(f"MDA file decided on: {ModelVars.mda_path}")

            if ModelVars.mda_path.is_file():
                print(f"✅ Found MDA: {ModelVars.mda_path}")
                ModelVars.grouped_maps = parse_material_file(str(ModelVars.mda_path))

                mda_texture_rel_path_str = None
                if ModelVars.grouped_maps:
                    if len(ModelVars.grouped_maps) > 1:
                        ModelVars.multiple_profiles = True
                    else:
                        ModelVars.multiple_profiles = False

                    ModelVars.map_key_to_use = "DFLT" if "DFLT" in ModelVars.grouped_maps else next(iter(ModelVars.grouped_maps), None)
                    # print(f"\nGROUPED MAPS: {ModelVars.grouped_maps}")
                    print(f"MAP KEY TO USE: {ModelVars.map_key_to_use}\n")

                    if ModelVars.map_key_to_use and ModelVars.map_key_to_use in ModelVars.grouped_maps:
                        texture_list_for_key = ModelVars.grouped_maps[ModelVars.map_key_to_use]
                        # print(f"TEXTURE LIST FOR KEY: {texture_list_for_key}")

                        if isinstance(texture_list_for_key, list) and 0 <= skin_index < len(texture_list_for_key):
                            mda_texture_rel_path_str = texture_list_for_key[skin_index]
                        else:
                            mda_texture_rel_path_str = texture_list_for_key[0]
                            print(f"⛔ Warning: Skin index {skin_index} out of bounds or invalid format for key '{ModelVars.map_key_to_use}' in MDA maps.")
                            print("⛔ Using first texture anyway (Models like Rictus refer to the same texture twice) ⛔")
                    else:
                        print("⛔ Warning: No suitable key ('DFLT' or first key) found in MDA maps or map is empty.")
                else:
                    print("⛔ Warning: MDA file parsed to empty grouped_maps.")

                if mda_texture_rel_path_str:
                    print(f"MDA relative path: {mda_texture_rel_path_str}")
                    current_texture_search_base_str = merge_paths(str(initial_model_dir), str(mda_texture_rel_path_str))
                    print(f"ℹ️ Path after MDA merge: {current_texture_search_base_str}")

                    mda_derived_tex_base = Path(current_texture_search_base_str)
                    print(f'ℹ️ Texture Path (from MDA): {mda_derived_tex_base} ({",".join(SUPPORTED_IMAGE_FORMATS)})')
                    found_texture_file_path, found_texture_resolution = _try_load_texture_from_base(
                        mda_derived_tex_base, SUPPORTED_IMAGE_FORMATS
                    )

                    if found_texture_file_path:
                        ModelVars.texture_paths[skin_index] = str(found_texture_file_path)
                        ModelVars.skin_resolutions[skin_index] = found_texture_resolution
            else:
                print(f"❌ No MDA file found for {model_filename_stem} in {initial_model_dir} or {initial_model_dir.parent}")


        # --- ATD Fallback ---
        if ModelVars.texture_paths[skin_index] == "":
            print("❌ Unable to locate texture after MDA (or MDA not found/used), trying ATD...")
            atd_file_candidate = Path(current_texture_search_base_str).with_suffix(".atd")
            print(f'ℹ️ ATD file lookup: {atd_file_candidate}')

            if atd_file_candidate.is_file():
                print(f"✅ Found ATD: {atd_file_candidate}")
                res_from_atd = extract_file_value(str(atd_file_candidate))
                if res_from_atd:
                    print(f"ℹ️ ATD extraction result: {res_from_atd}")
                    current_texture_search_base_str = merge_paths(current_texture_search_base_str, res_from_atd)
                    print(f"ℹ️ Path after ATD merge: {current_texture_search_base_str}")

                    atd_derived_tex_base = Path(current_texture_search_base_str)
                    print(f'ℹ️ Texture Path (from ATD): {atd_derived_tex_base}({",".join(SUPPORTED_IMAGE_FORMATS)})')
                    found_texture_file_path, found_texture_resolution = _try_load_texture_from_base(
                        atd_derived_tex_base, SUPPORTED_IMAGE_FORMATS
                    )
                    if found_texture_file_path:
                        ModelVars.texture_paths[skin_index] = str(found_texture_file_path)
                        ModelVars.skin_resolutions[skin_index] = found_texture_resolution
                else:
                    print(f"❌ ATD file {atd_file_candidate} did not yield a value.")
            else:
                print(f"❌ ATD file {atd_file_candidate} not found.")

        # Final check if texture was found for this skin_index
        if ModelVars.texture_paths[skin_index] == "":
            print(f"❌ Unable to locate texture for '{current_texture_search_base_str}/{embedded_texture_name}'!")
            # Ensure defaults are set if all attempts failed
            ModelVars.skin_resolutions[skin_index] = (64, 64)
            ModelVars.texture_paths[skin_index] = ""


    print("\n")
    print(f"Skin resolution info:\n{ModelVars.skin_resolutions}")






def get_md2_object():
    load_texture_coordinates(ModelVars.byte_list[ModelVars.header.ofs_st:ModelVars.header.ofs_tris], ModelVars.header, ModelVars.triangles, ModelVars.skin_resolutions, ModelVars.triangle_skin_dict, ImportOptions.texture_scale)

    for i_frame in range(len(ModelVars.frames)):
        for i_vert in range((ModelVars.header.num_xyz)):
            ModelVars.frames[i_frame].verts[i_vert].v[0] = ModelVars.frames[i_frame].verts[i_vert].v[0] * ModelVars.frames[i_frame].scale.x + ModelVars.frames[i_frame].translate.x
            ModelVars.frames[i_frame].verts[i_vert].v[1] = ModelVars.frames[i_frame].verts[i_vert].v[1] * ModelVars.frames[i_frame].scale.y + ModelVars.frames[i_frame].translate.y
            ModelVars.frames[i_frame].verts[i_vert].v[2] = ModelVars.frames[i_frame].verts[i_vert].v[2] * ModelVars.frames[i_frame].scale.z + ModelVars.frames[i_frame].translate.z

    ModelVars.vertices = list()
    for i, vert in enumerate(ModelVars.frames[0].verts):
        ModelVars.vertices.append(vertex_indexed(i, vert))

    ModelVars.my_object = md2_object(ModelVars.header, ModelVars.skin_names, ModelVars.triangles, ModelVars.frames, ModelVars.texture_coordinates, ModelVars.gl_commands, ModelVars.tsurf_dictionary, ModelVars.vertices, ModelVars.triangle_skin_dict, ModelVars.extra_data, ModelVars.texture_paths)









    # Gets name to give to the object and mesh in the outliner
    if not ImportOptions.displayed_name:
        # Added support for autoloading textures and to name mesh the same as filename if a Display Name is not entered on import screen - Creaper
        ModelVars.object_name = os.path.basename(ImportOptions.filepath).split('/')[-1]
        ModelVars.object_name = os.path.splitext(ModelVars.object_name)[0] # remove extension (last one)
        print(f"Blender Outliner Object Name: {ModelVars.object_name}")
        ModelVars.mesh = bpy.data.meshes.new(ModelVars.object_name)  # add the new mesh via filename
    else:
        ModelVars.object_name = [ImportOptions.displayed_name]
        print(f"Blender Outliner Object Name: {ImportOptions.displayed_name}")
        ModelVars.mesh = bpy.data.meshes.new(*ModelVars.object_name)  # add the new mesh, * extracts string from Display Name input list

    # List of vertices [x,y,z] for all frames extracted from the md2 object
    ModelVars.all_verts
    ModelVars.all_verts = [[x.v for x in ModelVars.my_object.frames[y].verts] for y in range(ModelVars.my_object.header.num_frames)]


def create_mesh_md2():
    print("Creating mesh...")

    # List of vertex indices forming a triangular face
    tris = ([x.vertexIndices for x in ModelVars.my_object.triangles])

    # uv coordinates (in Quake II/OpenGL terms st coordinates) for projecting the skin on the model's faces
    # blender uv coordinate system originates at lower left,
    # blender flips images upside down when loading so v = 1-t for blender imported images
    uvs_others = ([(x.s, 1-x.t) for x in ModelVars.my_object.texture_coordinates])

    ModelVars.obj = bpy.data.objects.new(ModelVars.mesh.name, ModelVars.mesh)

    # col = bpy.data.collections.get("Collection")  # To specify a specific collection
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
            # Paco WTF UV fix
            if 'paco' in ModelVars.object_name.lower() and vert_idx in (9,289):
                print(f"Fixing Paco UV vertex: {vert_idx}!")
                uv_layer.data[loop_idx].uv = [0.016129032258064516, 0.6587301587301587]
            else:
                uv_layer.data[loop_idx].uv = uvs_others[ModelVars.my_object.triangles[face_idx].textureIndices[idx]]



    """ Create animation for animated models: set keyframe for each vertex in each frame individually """
    # Frame names follow the format: amb_a_001, amb_a_002, etc...
    # A new animation will have a new prefix
    # bpy.context.area.type = "NLA_EDITOR"
    # bpy.context.area.ui_type = "NLA_EDITOR"

    ModelVars.animation_list = list()
    ModelVars.current_anim_name = ""
    frame_count = len(ModelVars.my_object.frames)

    bpy.ops.wm.import_animation_frames_modal('INVOKE_DEFAULT')
    # bpy.ops.wm.import_animation_frames('EXEC_DEFAULT')


    bpy.ops.wm.import_materials('EXEC_DEFAULT')







