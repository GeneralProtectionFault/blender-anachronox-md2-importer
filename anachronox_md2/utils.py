import bpy
from dataclasses import dataclass, fields
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

    texture_count: int                  # number of different textures the model uses
    ofs_glcmd_counts: int               # offset to the counts of # of gl commands for each texture
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


def load_import_variables(filepath, displayed_name, model_scale, texture_scale, x_rotate, y_rotate, z_rotate, apply_transforms, recalc_normals, use_clean_scene):
    ImportOptions.filepath = filepath
    ImportOptions.displayed_name = displayed_name
    ImportOptions.model_scale = model_scale
    ImportOptions.texture_scale = texture_scale
    ImportOptions.x_rotate = x_rotate
    ImportOptions.y_rotate = y_rotate
    ImportOptions.z_rotate = z_rotate
    ImportOptions.apply_transforms = apply_transforms
    ImportOptions.recalc_normals = recalc_normals
    ImportOptions.use_clean_scene = use_clean_scene


class ModelVars(object):
    """
    Class used to store what we need to process the model.
    Essentially a global object so we don't have confusion between Operators, etc...
    """
    my_object = {}
    obj = {}
    animation_list = list()
    all_verts = []
    current_anim_name = ""
    md2_path = ""
    model_path = ""
    mesh = {}
    object_name = ""

    byte_list = None
    header : md2_t
    skin_names = []
    model_filename = ""
    triangles = []
    frames = []
    gl_commands = []
    vertices = []
    texture_coordinates = []
    extra_data = None
    triangle_skin_dict = {}

    tsurf_dictionary = {}

    texture_paths = []
    skin_resolutions = {}

    grouped_maps = {}
    map_key_to_use = None
    mda_path = ""

    multiple_profiles = False


class ImportOptions(object):
    filepath = ""
    displayed_name = ""
    model_scale = 1.0
    texture_scale = 1.0
    x_rotate = 0.0
    y_rotate = 0.0
    z_rotate = 0.0
    apply_transforms = True
    recalc_normals = True
    use_clean_scene = True


# def startProgress(string):
#     print(string)
#     wm = bpy.context.window_manager
#     wm.progress_begin(0, 100)

# def endProgress():
#     wm = bpy.context.window_manager
#     wm.progress_update(100)
#     wm.progress_end()

# def showProgress(n, total, string=None):
#     pct = (100.0*n)/total
#     wm = bpy.context.window_manager
#     wm.progress_update(int(pct))
#     if string:
#         print(string)


def findnth(string, substring, n):
    """
    Find the nth instance of a substring and return the index
    """
    parts = string.split(substring, n)
    if len(parts) <= n:
        return -1
    return len(string) - len(parts[-1]) - len(substring)


def get_blender_area(area_type):
    for screen in bpy.context.workspace.screens:
        for area in screen.areas:
            if area.type == area_type:
                return area
    return None

'''
# TEST - The correct resulting vector is [1174, 513, 1185]

bytes = 2486176918      # BINARY - 10010100001100000000110010010110
                        # First 11 (as masked below):   10010010110 (1174)

print(0 & 0x000007ff)   # BINARY - 11111111111

# Bitshift 11
#print(11 & 0x000003ff)  # BINARY - 1111111111
#print(21 & 0x000007ff)  # BINARY - 11111111111


print (bytes >> 11)
'''
