#!/usr/bin/env python

"""
==================================
Preprocessing: Create Raster Tiles 
==================================

 ----- PURPOSE ----- 
This script takes large raster files in the form of '.tif' or '.img' and breaks them into tiles of a specified pixel length (default = 150 pixels by 150 pixels). The output of the script is a folder of these cropped tiles, with their colour channels corrected into 3 channels. These can be fed into the CNN Neural Network. 

The tile size (pixel length) is a changeable parameter that can be ammended to suit the size you need. To make use of this functionality, use the 'imageHeight' and 'imageWidth' parameters described below. 

NB: The script requires the 'Preprocessing_Utils.py' script found in the utils folder. 


 ----- OPTIONAL PARAMETERS -----
 There are 9 parameters (3 = required, 6 = optional):
 
    -o   --outDirectory   <str>    Path to where the tile images should be stored (required) 
    -r   --rasterFile     <str>    Path to where the '.tif' or '.img' raster file is stored (required)
    -c   --crsCode        <str>    The original crs code of the raster image (required)
    -h   --imageHeight    <str>    The pixel size which the tile's height should be
    -w   --imageWidth     <int>    The pixel size which the tile's width should be 
    -ic  --imageClass     <str>    Differentiation parameter to place images in the desired folder structure
    -op  --outPrefix      <str>    The filename pattern to save the output tile images with
    -rgb --rgbPrefix      <str>    The filename pattern to save the colour-corrected tile images with
    -b   --rgbBands       <list>   The red, green, and blue colour bands to extract from the raster image
       

 ----- USAGE ----- 
$ python3 proj/DataAnalysis/Pre-Processing/src/Preprocessing.py --outDir '/work/proj/DataAnalysis/Pre-Processing/out/' --rasterFile '/work/proj/TRAP_Data/kaz_fuse.img' --crsCode 'epsg:32635'

 -----  CREATED/MODIFIED -----
 Created: 26 January 2022 by Orla Mallon

""" 

"""
-----------------------
Burial Mound Defaults:
-----------------------
outDir = '/work/proj/DataAnalysis/Pre-Processing/out/'
crsCode = 'epsg:32635'
rasterFile = '/work/proj/TRAP_Data/kaz_fuse.img'
"""


#     ===============================     SCRIPT     ===============================

"""
=======================
Import the Dependencies
=======================

"""
# --- Operating system ---
import os
import sys
import argparse

# --- Data handling packages ---
import numpy as np      
import matplotlib.pyplot as plt
from PIL import Image

# --- Geospatial packages --- 
import rasterio
from rasterio import features
from rasterio.mask import mask
from rasterio.transform import TransformMethodsMixin
from rasterio.coords import BoundingBox
from rasterio.windows import Window
import rasterio.plot as rplt
from shapely.geometry import Polygon
from shapely.geometry import Point
import geopandas as gpd

# --- Functions from the utils folder ---
import utils.Create_Tiles_Utils as functions

"""
=============
Main Function
=============
"""
def main():
    
    """
    ---- Argparse Parameters ----
    """
        
    # Initialize ArgumentParser class
    ap = argparse.ArgumentParser()
    
    # Argument 1: Path to the out directory
    ap.add_argument('-o',    '--outDirectory',    type = str,    required = True,    nargs=1,
                help = 'Path to where the tile images should be stored. Needs a '/' at the end.\n\n')
    
    # Argument 2: Path to the raster image 
    ap.add_argument('r', '--rasterFile',    type = str,    required = True,    nargs=1,
                    help = "Path to where the '.tif' or '.img' raster file is stored\n\n")
    
    # Argument 3: The original crs code of the raster image
    ap.add_argument('-c',    '--crsCode',    type = str,    required = True,    nargs=1,
                    #default = 'epsg:32635', 
                    help = "The CRS code to use. Must have lowercase letters and numbers.\n
                    [IMPORTANT] The crs_code must be wrapped in quotation marks (e.g., ' ', or " ")\n
                    [EXAMPLE] 'epsg:32635'\n")
    
    # Argument 4: The tile height i.e., pixel size which the tile's height should be
    ap.add_argument('-h',    '--imageHeight',    type = int,    required = False,    nargs=1,
                    help = "The size the cutout tile's height should be in pixels. Default = 150px. \n",
                    default = 150)
    
    # Argument 5: The tile width i.e., pixel size which the tile's width should be
    ap.add_argument('-w',    '--imageWidth',    type = int,    required = False,    nargs=1,
                    help = "The size the cutout tile's width should be in pixels. Default = 150px. \n",
                    default = 150)
    
    # Argument 6: The imageClass differentiator to use (i.e., East, west?) 
    ap.add_argument('-ic',    'imageClass',    type = str,    required = False,    nargs=1,
                    help = "Differentiation parameter to place images in the desired folder structure.\n
                            [IMPORTANT] The imageClass must be wrapped in quotation marks (e.g., ' ', or " ")\n
                            [Example] 'east150' or 'west150' for images from the desired region of the image.",
                    default = 'tiles{:d}'.format(imageSize))
                    
    # Argument 7: The outPrefix filename pattern to save the tile images with
    ap.add_argument("-op",    "--outPrefix",    type = str,    required = False,    nargs=1,
                    help = "The filename pattern to save the tile images with.\n
                    [EXAMPLE] IMG_{:.4f}_{:.4f}_{:d}.format(x, y, imageSize) ",
                    default = None)
                    
    # Argument 8: The rgbPrefix filename pattern to save the tile images with 
    ap.add_argument("-rgb",    "--rgbPrefix",    type = str,    required = False,    nargs=1,
                    help = "The filename pattern to save the colour-corrected tile images with.\n",
                    default = None) 
                    
    # Argument 9: The rgb colour band channels to use from the raster - needs to be red, green and blue 
    ap.add_argument("-b",    "--rgbBands",    type = list,    required = False,    nargs=1, 
                    help = "The red, green & blue colour bands to extract from the raster image.\n
                            [Default] [4, 3, 2]\n",
                    default = [4, 3, 2])
                    
    # Parse arguments
    args = vars(ap.parse_args())
    

    """
    ---- Fit the variables to the parameters ----
    """
    #Args arguments
    
    # --- Filepath parameters ---
    outDir = args["outDirectory"]           # Argument 1
    outPrefix = ["outPrefix"]               # Argument 7 
    rgbPrefix = args["rgbPrefix"]           # Argument 8 
                    
        
    # --- Project parameters ---
    rasterPath = args["rasterFile"]         # Argument 2
    crsCode = args["crsCode"]               # Argument 3
    imageHeight = args["imageHeight"]       # Argument 4
    imageWidth = args["imageWidth"]         # Argument 5
    imageClass = args["imageClass"]         # Argument 6
    rgbBands = args["rgbBands"]             # Argument 9
    
                
    
    """
    -------------------------------
    Step 1: Get the tile dimensions 
    -------------------------------

    """
                    
    # --- Open the raster image ---
    rasterData = rasterio.open(rasterPath)
    
    # --- Get tile dimensions ---   
    x, y, tile_dimensions = functions.extract_tile_dimensions(rasterData, 
                                                              imageHeight = imageHeight, 
                                                              imageWidth = imageWidth,
                                                              crs_code = crsCode)
                    
    # --- Save the tile dimensions ---
    #Create output directory for the GeoDataFrames 
    if not os.path.exists(outDir + "GeoDataFrames/"):
        os.makedirs(outDir + "GeoDataFrames/")
    
    #Save the GeoDataFrame as a csv file 
    csv_path = outDir + "/" + "GeoDataFrames" + "/" + "tile_dimensions.csv"
    tile_dimensions.to_csv(csv_path)
                    
    """
    ----------------------
    Step 2: Crop the tiles
    ----------------------

    """
    imageSize = imageHeight    # note you will need to ammend this if you are using rectangle tiles
    
    for index, row in tile_dimensions.iterrows():
        
        x = row["geometry"].x
        y = row["geometry"].y
    
        functions.crop_tile(rasterData = rasterData, 
                            x = x, 
                            y  = y, 
                            outPrefix=outPrefix, 
                            imageSize=imageSize,
                            imageClass = imageClass)
        
    """
    ---------------------------------
    Step 3: Merge the colour channels  
    ---------------------------------

    """
    
    #Get the normalised zscores: beware this line takes a long time to run 
    zLow, zHigh = functions.get_zscores(rasterData)
    
    #For every row in tile_dimensions, merge the colour channels
    for index, row in tile_dimensions.iterrows():
        
        x = row["geometry"].x
        y = row["geometry"].y
    
        functions.merge_colour(rasterData = rasterData, 
                               x = x, 
                               y  = y,
                               imageClass = imageClass,
                               rgbPrefix= rgbPrefix, 
                               imageSize= imageSize, 
                               zLow = zLow,
                               zHigh = zHigh,
                               rgbBands= rgbBands) 


#Close the main function 
if __name__=="__main__":
    main() 