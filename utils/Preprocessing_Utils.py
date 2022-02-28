#!/usr/bin/env python

""" 
========================================================
PREPROCESSING UTILITIES SCRIPT: Cropping Spatial Rasters 
========================================================

This script contains 5 functions necessary for running the 'Preprocessing.py' script:

1. extract_tile_dimensions()
2. generate_polygon()
3. crop_tile()
4. get_zscores()
5. merge_colour()

"""

"""
------------
Dependencies
------------
"""
# --- Operating system ---
import os
import sys

# --- Data handling packages ---
import numpy as np      
import matplotlib.pyplot as plt
import glob
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


"""
----------------------------------------------------------
1. Function for extracting the numpy style tile dimensions 
----------------------------------------------------------
"""

def extract_tile_dimensions(rasterData, crs_code, imageSize):
    """
    A function to extract the rows, columns, bounds, and centroids of the tiles to be created.
    
    Returns a GeoDataFrame object containing this information.
    """
    
    """
    Set-up the lists and counters 
    """
    # ------------------ Create image and dimension counters ---------------------------
    #dynamic image counters 
    image_counter = 0
    image_idx = []
    
    #dynamic dimension counters 
    #to keep track of the moving sides 
    col_off = 0
    row_off = 0
    
    
    # ------------- Create lists to store the pixel rows, columns, and  x/y coordinates of image ------
    
    #row and column of the tile's upper left corner
    row_upper_left = []
    column_upper_left = []
    
    #row and column of the tile's centre point 
    row_centre = []
    column_centre = []
    
    #The x and y coordinates of the tile's centre point
    x_coordinate = []
    y_coordinate = []
    
    
    # ------------------ Define the static width and height of the cutout tile --------------------
    width = imageSize
    height = imageSize
    
    
    """
    Get the number of tiles that fit into the height and width dimensions of the raster image
    """
    
    # ------------------ HEIGHT ----------------------------
    # Get the height of the image 
    no_boxes_height = rasterData.shape[0]
    
    #divide it by the height of the tiles 
    no_boxes_height = no_boxes_height / height
    
    #round decimal down to integer, to ensure all tiles fit within the raster image
    no_boxes_height = int(no_boxes_height) 
    
    
    # ------------------ WIDTH ---------------------------- 
    # Get the height of the image 
    no_boxes_width = rasterData.shape[1]
    
    #divide it by the height of the tiles 
    no_boxes_width = no_boxes_width / width
    
    #round decimal down to integer, to ensure all tiles fit within the raster image
    #python will always round down 
    no_boxes_width = int(no_boxes_width) 
    
    
    """
    Loop through the image and use the dimensions to extract x and y coordinates for each tile 
    """
    
    # ------------------ Step 1: Find the row and column values of our upper left corners -------------
    
    ### i represents the height of the image, where we know we can fit 'no_boxes_height' down 
    ### j represents the width of the image, where we know we can fit 'no_boxes_width' across
    ### Note this style means the pixels outside the last boxes on each side will be excluded from tiles
    for i in range(0, no_boxes_height):
        for j in range(0, no_boxes_width):
            #update counters and image index 
            image_counter = image_counter + 1
            image_idx.append(image_counter)
            
            #Get the row and columns 
            row_off = i * 150
            col_off = j * 150 
        
            #save these as the upper left corner of image 
            column_upper_left.append(col_off)
            row_upper_left.append(row_off)
            
            # ------------------ Step 2: Get the center pixel values of the window -------------------
            centre_x = int(col_off + width/2)
            centre_y = int(row_off + height/2)
        
            #Save the row and column of tile's centre point
            column_centre.append(centre_x) 
            row_centre.append(centre_y)
            
            # ------------------ Step 3: Transform the centre row/column into x and y points ----------
            x, y = rasterData.transform * (centre_x, centre_y)
            x_coordinate.append(x)
            y_coordinate.append(y)
    
    print(f"Finished getting dimensions! you have {image_counter} images") 
            
    """
    Create a spatial GeoDataFrame  
    """
    
    # ------------------ Create a spatial point ----------------------------------
     
    #First, create a tuple with our x and y coordinates 
    geometry = list(zip(x_coordinate, y_coordinate))
    
    #Next, make the geometry tuple a spatial point
    #Create empty list to store the points in 
    Points = []

    #Loop through the geometry column and transform the tuple to a spatially recognised point
    for i in geometry:
        pnt = Point(i[0], i[1])
        Points.append(pnt)

    #Check format
    print(f"\n[Data-check] The first point has the following coordinates: {Points[0]} \n")
    
    
    # ------------------ Ensure all points are within the raster area ------------------
    
    #Create a list of raster_bound_coordinates using generate_polygon() function (defined below)
    raster_bound_coordinates = generate_polygon(rasterData.bounds)
    print(f"[Data-check] The raster's bounds are: {raster_bound_coordinates}\n")
    
    #Create a polygon and check all points fall within it 
    raster_area = Polygon([(raster_bound_coordinates[0]), (raster_bound_coordinates[1]), 
                           (raster_bound_coordinates[2]), (raster_bound_coordinates[3])])
    
    #Loop through the points and ensure they're all within the bounds
    #If nothing is printed that means all images are within the bounds
    for pnt in Points:
        if pnt.within(raster_area):
            pass
        else: 
            print(f"\nPoint of coordinates: {pnt} is not in the bounding box.\n") 
            
            
    # ------------------ Put all information into GeoDataFrame -------------------------
    
    #Create geodataframe 
    tile_dimensions = gpd.GeoDataFrame({
        
        'image_idx': image_idx, 
        'row_upper_left': row_upper_left,
        'column_upper_left': column_upper_left,
        'row_centre': row_centre,
        'column_centre': column_centre,
        'x_coordinate': x_coordinate,
        'y_coordinate': y_coordinate,
        'geometry': Points}) 
    
    #Reset the index to the image number 
    tile_dimensions = tile_dimensions.set_index('image_idx')

    # ------------------ reset the CRS -------------------------
    
    #Check the crs of the original data 
    print(f"\nThe original raster's CRS is: {rasterData.crs}\n")
    
    #Set the tile_dimensions crs to the desired crs 
    tile_dimensions = tile_dimensions.set_crs(crs_code)
    
    #Check the new CRS 
    print(f"The tile_dimensions CRS is now {tile_dimensions.crs}. This should match the original raster's CRS above.")
    
    print("\nYour information is saved as a GeoDataFrame.")
    
    return x_coordinate, y_coordinate, tile_dimensions

"""
--------------------------------------------------------------------------
2. Function to get the spatial bounds (coordinates) of tile's bounding box 
--------------------------------------------------------------------------
"""

 def generate_polygon(bbox):
    """
    Generates a list of coordinates: [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x1,y1]]
    
    Returns a list of tuples
    """
    return [[bbox[0],bbox[1]],
            [bbox[2],bbox[1]],
            [bbox[2],bbox[3]],
            [bbox[0],bbox[3]],
            [bbox[0],bbox[1]]]

"""
--------------------------------------------------------
3. Function to crop the tile into a subset of the raster 
--------------------------------------------------------
"""

#

def crop_tile(rasterData, x, y, imageClass, outPrefix=None, imageSize=150):

    """
    A function to crop the raster into the desired tile dimensions using rasterio's mask() function. 
    
    Output:  the tile images saved to the chosen directory. 
    """
    
    #----------- Get the min and max values of our tile's bounding point, using the centre point(x,y) ---------
   
    #Round the x and y values
    x = round(x+0.5)-0.5
    y = round(y+0.5)-0.5
    
    #Get the bounds of the tile 
    xMin = float(x - imageSize/2.)
    yMin = float(y - imageSize/2.)
    xMax = float(x + imageSize/2.)
    yMax = float(y + imageSize/2.)
    
    """
    Create a GeoJSON dict object
    """ 
    # This is the format necessary for the rasterio package to mask the image (ie crop it) 
    #There are 5 sides as it needs to be a 'closed' polygon
    polyCoords = [[[xMax, yMin], [xMax, yMax], [xMin, yMax], 
                   [xMin, yMin], [xMax, yMin]]]
    
    polyDict = {'type': 'Polygon', 
                'coordinates': polyCoords}
        
    """
    Crop the image from the main raster using mask 
    """ 

    # Now use the mask function of rasterio to clip the data
    outData, outTrans = mask(dataset=rasterData, 
                             shapes=[polyDict],
                             all_touched=True, 
                             crop=True)
                
    """
    Update the metadata  
    """ 
    # Copy the metadata from the original file and modify to reflect changes
    # We are not changing CRS, so don't need to update
    outMeta = rasterData.meta.copy()
    outMeta.update({"driver": "GTiff",
                    "height": outData.shape[1],
                    "width":  outData.shape[2],
                    "transform": outTrans})
    
            
    """
    Format the filename for the image  
    """ 

    # Format the output file name and create the output directory(s)
    imageClass = imageClass
    
    if outPrefix is None:
        outPrefix = "IMG_{:.4f}_{:.4f}_{:d}".format(x, y, imageSize)
    if not os.path.exists(outDir + imageClass + "/"):
         os.makedirs(outDir + imageClass + "/")
    outPath = outDir  + imageClass + "/" + outPrefix + ".tif"
    
    """
    Save the cropped image as .tif file 
    """ 
    if os.path.exists(outPath):
        os.remove(outPath)
    with rasterio.open(outPath, "w", **outMeta) as FH:
        FH.write(outData)
        
        
"""
--------------------------------------
4. Get zScores for min-max normalising 
--------------------------------------
"""

def get_zscores(rasterData):
    """
    A function to get zLow and zHigh values of the original raster image based on each colour band's min and 
    max values. This is used for normalising the data. 
    
    Returns: two scores, zLow and zHigh
    """
    # --------------- Create placeholder variable for the min and max value ---------------
    
    zLow = None     
    zHigh = None  
    
    # --------------- Create empty list for storing the band information ---------------
    
    zMinLst = []
    zMaxLst = []
    zStdLst = []
    zMedLst = []
    
    # --------------- Loop through each band and find the min/max/std/median values ---------------
    
    #We're adding 1 to the counter because python starts indexing at 0
    for iBand in range(1, rasterData.count+1):
        print(" {:d} ...".format(iBand), end="", flush=True)
        bandArr = rasterData.read(iBand)       
        zMinLst.append(np.nanmin(bandArr))             #returns minimum of an array while ignoring Na's
        zMaxLst.append(np.nanmax(bandArr))             #returns maximum of an array while ignoring Na's
        zStdLst.append(np.nanstd(bandArr))             #returns std of an array while ignoring Na's
        zMedLst.append(np.nanmedian(bandArr))          #returns median of an array while ignoring Na's
        
    # --------------- Find the min/max/std/median values from the lists --------------- 
    
    zMin = np.min(zMinLst)
    zMax = np.max(zMaxLst)
    zStd = np.max(zStdLst)
    zMed = np.max(zMedLst)
    
    print("[INFO] Range = ({:.1f}, {:.1f}), Med = {:.1f}, Stdev = {:.1f}"\
          .format(zMin, zMax, zMed, zStd))
    
    # --------------- Save as variables ---------------
    
    zLow = np.max([0.0, zMin])     #clip at 0 
    zHigh = zMed + zStd * 5.        #clip at 5 stds 
    
    print(f"[INFO] zLow = {zLow}, and zHigh = {zHigh}")
    
    return zLow, zHigh 


"""
------------------------------------------------
5. A function to merge the 4 colour bands into 3
------------------------------------------------
"""

def merge_colour(rasterData, x, y, imageClass, rgbPrefix = None, imageSize=150, rgbBands=[4, 3, 2],
                 zLow=None, zHigh=None):
    
    """
    A function to merge the 4 colour bands into 3 
    
    Output: Tiles are saved as new .png images into the indicated RGB output directory
    """

    # --------------- Get the min and max values of our tile's bounding point, using the centre point(x,y) -----
   
    #Round the x and y values
    x = round(x+0.5)-0.5
    y = round(y+0.5)-0.5
    
    #Get the bounds of the tile 
    xMin = float(x - imageSize/2.)
    yMin = float(y - imageSize/2.)
    xMax = float(x + imageSize/2.)
    yMax = float(y + imageSize/2.)
    

    # --------------- Create a GeoJSON dict object ------------------------
    
    # This is the format necessary for the rasterio package to mask the image (ie crop it) 
    #There are 5 sides as it needs to be a 'closed' polygon
    polyCoords = [[[xMax, yMin], [xMax, yMax], [xMin, yMax], 
                   [xMin, yMin], [xMax, yMin]]]
    polyDict = {'type': 'Polygon', 'coordinates': polyCoords}
        
    # --------------- Crop the image from the main raster using mask ------------------------ 

    # Now use the mask function of rasterio to clip the data
    outData, outTrans = mask(dataset=rasterData, 
                             shapes=[polyDict],
                             all_touched=True, 
                             crop=True)
        
    # --------------- Create an RGB image with 3 colour bands ------------------------  
    
    #Create an RGB outPrefix 
    if rgbPrefix is None:
        rgbPrefix = "RGB_{:.4f}_{:.4f}_{:d}".format(x, y, imageSize)
    
    #Link up the directories 
    if not os.path.exists(outDir + "RGB/" + imageClass + "/"):
        os.makedirs(outDir + "RGB/" + imageClass + "/")
        
    # Scale the data to the range 0-255 for RGB values
    outR = (outData[rgbBands[0]-1] - zLow) * 255.0 / (zHigh -zLow)
    outG = (outData[rgbBands[1]-1] - zLow) * 255.0 / (zHigh -zLow)
    outB = (outData[rgbBands[2]-1] - zLow) * 255.0 / (zHigh -zLow)
    
    # Convert to 16-bit and stack into a 3D array
    outR = outR.astype(np.uint8)
    outG = outG.astype(np.uint8)
    outB = outB.astype(np.uint8)
    rgbArr = np.dstack((outR, outG, outB))
    
    # Write as a PNG file
    img = Image.fromarray(rgbArr)
    outRBGpath = outDir + "RGB/" + imageClass + "/"  + rgbPrefix + ".png"
    img.save(outRBGpath)
    
    
# Define behaviour when called from command line
if __name__=="__main__":
    pass