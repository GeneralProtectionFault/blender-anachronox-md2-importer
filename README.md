# Anachronox MD2 import add-on for Blender 3.x

## Based on:
https://github.com/lennart-g/blender-md2-importer

## Overview
In Blender, go to Edit => Preferences, then select Add-ons.  Click "Install" and select the .py file.  This will install the plugin.
To use, go to File => Import, then select Anachronox Model Import (.md2), then select the model file.

## Prerequisites
The original Anachronox models are stored in anoxdata\MODELS.dat, as found in the installed game folder.
They can be extracted with the DATExtract tool as found on the Anachrodox website:  http://anachrodox.talonbrave.info/
(Tools section)

Direct link to the utility (right-click & "Save Link As"):
http://anachrodox.talonbrave.info/tools/tools/datextract2.zip


Update:
Plugin will now support loading models with multiple textures.  A material is created per texture and correctly assigned to the appropriate parts of the model.

![image](https://user-images.githubusercontent.com/29645865/210277081-265c5ab1-16d2-4cec-9808-503561bb80a7.png)

## Information
The original model format (.MD2) is from Quake.  Anachronox uses a modified version of this model format, the critical difference being support for using multiple (in the real world, it does not appear they ever used more than 2) textures for 1 model.

The original Quake MD2 format does support multiple *skins* which are stored in the header.  These are simply the names of the texture images.  This can be confusing, but the original purpose was to have interchangeable texture images that would essentially act as a palette swap.  This is repurposed in the Anachronox format, so that multiple textures can be applied to a single model simultaneously--this being the crux of the incompatibility between the Quake & Anachronox models.

This is done by a few bytes that define 2 numbers (2 short integers between the end of the GL Commands and the Tagged Surfaces), a primitive offset and number of primitives.  "Primitives," in this case, refers to OpenGL (or just "gl") commands.  When actually drawing a model in OpenGL, the vertices/polygons are drawn in order according to the gl commands (i.e. draw a triangle with such-and-such vertices, etc...).  When the "nth" gl command is reached corresponding to the aforementioned "primitive offset" number, the model draws using the other texture for "n" gl commands.  Since Anachonox models only use 2, this seems to always bring us to the last gl command.
