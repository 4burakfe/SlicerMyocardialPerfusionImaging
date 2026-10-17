#-----------------------------------------------------------------------------
# Myocardial Perfusion SPECT hub: the only module in Slicer's module list. It shows the workflow
# steps below (hidden modules) and carries the helper package they all use.
set(MODULE_NAME MyocardialPerfusionSPECT)

set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${MODULE_NAME}Lib/__init__.py
  ${MODULE_NAME}Lib/geometry.py
  ${MODULE_NAME}Lib/myocardium.py
  ${MODULE_NAME}Lib/gating.py
  ${MODULE_NAME}Lib/resampling.py
  ${MODULE_NAME}Lib/views.py
  ${MODULE_NAME}Lib/colortables.py
  ${MODULE_NAME}Lib/display.py
  ${MODULE_NAME}Lib/studies.py
  ${MODULE_NAME}Lib/session.py
  ${MODULE_NAME}Lib/case.py
  ${MODULE_NAME}Lib/widgets.py
  ${MODULE_NAME}Lib/toolbar.py
  ${MODULE_NAME}Lib/parameters.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/Images/Banner.jpg
  )

slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
# Workflow steps (hidden modules, shown inside the hub)
foreach(STEP_MODULE MPIDataRoles MPIProcessing MPIMyocardium MPIReview MPIGatedQC MPIGated)
  slicerMacroBuildScriptedModule(
    NAME ${STEP_MODULE}
    SCRIPTS ${STEP_MODULE}.py
    WITH_GENERIC_TESTS
    )
endforeach()

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclasses in the module scripts as ctests.
  # Note that the tests will also be available at runtime.
  foreach(TEST_MODULE ${MODULE_NAME} MPIDataRoles MPIProcessing MPIMyocardium MPIReview MPIGatedQC MPIGated)
    slicer_add_python_unittest(SCRIPT ${TEST_MODULE}.py)
  endforeach()

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
