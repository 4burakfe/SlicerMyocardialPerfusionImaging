cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(MyocardialPerfusionSPECT)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/4burakfe/SlicerMyocardialPerfusionSPECT")
set(EXTENSION_CATEGORY "Nuclear Medicine")
set(EXTENSION_CONTRIBUTORS "Burak Demir")
set(EXTENSION_DESCRIPTION "Tarvos: myocardial perfusion SPECT and PET workflow. Study roles and case from DICOM, reorientation to the cardiac axes (SA, HLA, VLA), automatic LV wall detection, stress / rest slices, polar maps, 3D views and TID, gated QC and gated review with thickening and motion maps, LV volumes and EF. Works with volumes reconstructed by the DICOMSpectRecon extension.")
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
