
bl_info = {
    "name": "Anachronox MD2 Model Importer",
    "author": "Lennart G, Alpaca, Holonet, Creaper",
    "version": (1,0,1),
    "blender": (4,0,0),
    "location": "File > Import > Anachronox (.md2)",
    "description": "Import Anachronox variant of MD2 (Quake II) models",
    "warning": "",
    "github_url": "https://github.com/GeneralProtectionFault/blender-anachronox-md2-importer",
    "doc_url": ""
    }



"""
This part is required for the UI, to make the Addon appear under File > Import once it's
activated and to have additional input fields in the file picking menu  
Code is taken from Templates > Python > Operator File Import in Text Editor
The code here calls blender_load_md2
"""
# ImportHelper is a helper class, defines filename and
# invoke() function which calls the file selector.
from bpy_extras.io_utils import ImportHelper
from bpy.props import BoolProperty, StringProperty
import bpy
from .anachronox_md2_import import blender_load_md2
from .Processor import ImportAnimationFrames, ImportMaterials, QueueRunner


#----------------------------------------------------------
#   Register
#----------------------------------------------------------



class ImportSomeData(bpy.types.Operator, ImportHelper):
    """Loads a Quake 2 MD2 File"""
   
    # Added a bunch of nifty import options so you do not have to do the same tasks 1000+ times when converting models to another engine. -  Creaper

    bl_idname = "import_md2.some_data"  # important since its how bpy.ops.import_test.some_data is constructed
    bl_label = "Import MD2"

    ## ImportHelper mixin class uses this
    # filename_ext = ".md2"

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
                                        default=True)

    # Added option to clean the Blender scene of unused items do you don't end up with a bunch of stuff named .### and have to manually rename them
    use_clean_scene: BoolProperty(name="Clean Scene",
                                        description="Clean the Blender scene of any unused data blocks including unused Materials, Textures and Names.\nYou typically want this set.",
                                        default=True)

    
    def execute(self, context):
        try:
            return blender_load_md2(self.filepath, self.displayed_name, self.model_scale, self.texture_scale, self.x_rotate, self.y_rotate, self.z_rotate, self.apply_transforms, self.recalc_normals, self.use_clean_scene)
        except Exception as argument:
            self.report({'ERROR'}, str(argument))



def menu_func_import(self, context):
    self.layout.operator(ImportSomeData.bl_idname, text="Anachronox Model Import (.md2)")


classes = [
    ImportAnimationFrames,
    ImportMaterials,
    QueueRunner,
    ImportSomeData
]

def register():
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    
    for cls in classes:
        # print(f'Registering: {cls}')
        bpy.utils.register_class(cls)

    QueueRunner.define("WM_OT_import_animation_frames")
    QueueRunner.define("WM_OT_import_materials")


# called when addon is deactivated (removed script from menu)
def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

def missing_file(self, context):
    self.layout.label(text="Model file does not exist in currently selected directory! Perhaps you didn't select the correct .md2 file?")


if __name__ == "__main__":
    register()
    print("Anachronox MD2 Importer loaded.")
