import bpy


class ModelVars(object):
    my_object = {}
    obj = {}
    animation_list = list()
    all_verts = []
    current_anim_name = ""
    md2_path = ""
    model_path = ""
    mesh = {}
    x_rotate = 0 
    y_rotate = 0 
    z_rotate = 0
    apply_transforms = True
    recalc_normals = False
    object_name = ""


def startProgress(string):
    print(string)
    wm = bpy.context.window_manager
    wm.progress_begin(0, 100)

def endProgress():
    wm = bpy.context.window_manager
    wm.progress_update(100)
    wm.progress_end()

def showProgress(n, total, string=None):
    pct = (100.0*n)/total
    wm = bpy.context.window_manager
    wm.progress_update(int(pct))
    if string:
        print(string)


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