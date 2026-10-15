cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(MyocardialPerfusionSPECT)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/4burakfe/SlicerMyocardialPerfusionSPECT")
set(EXTENSION_CATEGORY "Nuclear Medicine")
set(EXTENSION_CONTRIBUTORS "Burak Demir")
set(EXTENSION_DESCRIPTION "Myocardial perfusion SPECT processing: interactive reorientation to the cardiac axes (short axis, horizontal and vertical long axis) with ASNC/SNMMI standard display, ellipsoid masking of extracardiac activity and a slice review panel. Works with volumes reconstructed by the DICOMSpectRecon extension.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/4burakfe/SlicerMyocardialPerfusionSPECT/main/MyocardialPerfusionSPECT/Resources/Icons/MyocardialPerfusionSPECT.png")
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(MyocardialPerfusionSPECT)

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
