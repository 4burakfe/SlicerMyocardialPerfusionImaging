"""Saving the exam: the scene as a bundle, screenshots, and the screen / the scene as DICOM.

- Save scene (.mrb): the whole scene (studies, reoriented volumes, myocardium, results, MPI settings) in one
  Medical Reality Bundle, named after the case, in the last folder used (Slicer's own Save dialog is offered too).
- Save screenshot (PNG): the views of the current layout (and the Monitor 2 window when it is shown).
- Save screen as DICOM: the same picture as a DICOM Secondary Capture (RGB) that every DICOM viewer and PACS
  shows, filed under the patient and study of the MPI studies.
- Save scene as DICOM: the whole scene zipped into a private tag of a secondary capture with a screenshot
  (3D Slicer's "Slicer Data Bundle", as in the Taranis extension). Slicer's DICOM browser recognises it and opens
  the scene; only Slicer reads it, and some PACS, routers and anonymisers drop private tags.

DICOM objects are written to a chosen folder and added to Slicer's DICOM database (from there they can be sent to a
PACS with the DICOM browser). They are filed under the study of the active study (else any study of the exam) that
came from DICOM; without one, under the case name and ID with a study UID kept on the MPI session node, so every
export of the scene lands in one study.
The pure functions at the top (PPM decoding, the Secondary Capture dataset) need numpy and pydicom only.
"""

import datetime
import logging
import os
import shutil
import tempfile

import numpy as np

# ---------------------------------------------------------------------------
# Settings of this file: edit the values here. A value shown in Advanced parameters takes the one here as its
# built-in default; a value saved with Save parameters overrides it until Restore defaults.
# ---------------------------------------------------------------------------
MANUFACTURER = "Tarvos (3D Slicer)"
SECONDARY_CAPTURE_STORAGE = "1.2.840.10008.5.1.4.1.1.7"
SCREEN_SERIES_NUMBER = 951
SCENE_SERIES_NUMBER = 952
SETTING_DICOM_FOLDER = "MyocardialPerfusionImaging/DicomExportFolder"
SETTING_SCENE_FOLDER = "MyocardialPerfusionImaging/SceneFolder"
SETTING_SCREENSHOT_FOLDER = "MyocardialPerfusionImaging/ScreenshotFolder"
EXPORT_STUDY_ATTRIBUTE = "MPI.ExportStudyUID"   # study of the exports when no study came from DICOM

# patient / study attributes copied from the reference DICOM file
STUDY_TAGS = ["PatientName", "PatientID", "PatientBirthDate", "PatientSex", "StudyInstanceUID", "StudyID",
              "StudyDate", "StudyTime", "StudyDescription", "AccessionNumber", "ReferringPhysicianName"]

TOOLTIPS = {
    "save": ("<b>Save</b><br>"
             "<b>Save scene (.mrb)</b>: the whole exam (studies, reoriented volumes, myocardium, results and the MPI "
             "settings) in one file named after the case; open it again with File &gt; Load scene or by dropping it "
             "on Slicer.<br>"
             "<b>Save screenshot (PNG)</b>: the views shown (and the Monitor 2 window when it is open)."),
    "screen": ("<b>Save screen as DICOM</b><br>"
               "Saves what the views show (slices, polar maps, 3D, curves; also the Monitor 2 window when it is "
               "open) as a standard <b>DICOM Secondary Capture</b> that every DICOM viewer and PACS displays.<br>"
               "It is filed under the <b>patient and study of the MPI studies</b>, written to a folder you choose "
               "and added to Slicer's DICOM database, from where it can be sent to a PACS.<br>"
               "<i>Research use only.</i>"),
    "scene": ("<b>Save scene as DICOM</b><br>"
              "Packs the <b>whole Slicer scene</b> (every study, the reoriented volumes, the myocardium, results and "
              "the MPI settings) into one DICOM file (3D Slicer's <i>Slicer Data Bundle</i>: a secondary capture with "
              "a screenshot, the scene zipped in a private tag).<br>"
              "It is filed under the <b>patient and study of the MPI studies</b>, written to a folder you choose and "
              "added to Slicer's DICOM database, from where it can be sent to a PACS.<br>"
              "To open it: load it from the DICOM browser; Slicer unpacks it and opens the scene, also on a computer "
              "that has none of the images.<br>"
              "<i>Only 3D Slicer can read it. Some PACS, routers and anonymisers drop private tags or refuse large "
              "secondary captures: check once that yours keeps it. Research use only.</i>"),
}


class ExportError(Exception):
    pass


# --- Pure helpers (numpy / pydicom) ---------------------------------------------------------------------------

def ppmPixels(data):
    """RGB pixels (rows, columns, 3; uint8) of a binary PPM (P6, maxval 255) file content."""
    fields, pos = [], 0
    while len(fields) < 4:
        while pos < len(data) and data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":   # comment up to the end of the line
            pos = data.index(b"\n", pos) + 1
            continue
        end = pos
        while end < len(data) and not data[end:end + 1].isspace():
            end += 1
        if end == pos:
            raise ExportError("The screenshot could not be read (truncated PPM header).")
        fields.append(data[pos:end])
        pos = end
    pos += 1   # one whitespace character after maxval
    magic, width, height, maxValue = fields[0], int(fields[1]), int(fields[2]), int(fields[3])
    if magic != b"P6" or maxValue != 255:
        raise ExportError(f"The screenshot could not be read (PPM {magic!r}, maxval {maxValue}).")
    pixels = np.frombuffer(data, dtype=np.uint8, count=width * height * 3, offset=pos)
    return pixels.reshape(height, width, 3)


def secondaryCaptureDataset(rgb, tags, seriesUID, instanceNumber, seriesDescription, comment=""):
    """A DICOM Secondary Capture dataset (pydicom) of an RGB image (rows, columns, 3; uint8). tags: patient /
    study attributes (a new StudyInstanceUID when missing)."""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ExportError("The screenshot is not an RGB image.")
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SECONDARY_CAPTURE_STORAGE
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.preamble = b"\0" * 128
    now = datetime.datetime.now()
    ds.SpecificCharacterSet = "ISO_IR 192"
    ds.SOPClassUID = SECONDARY_CAPTURE_STORAGE
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    for keyword in STUDY_TAGS:
        if tags.get(keyword):
            setattr(ds, keyword, tags[keyword])
    for keyword in ("PatientName", "PatientID", "PatientBirthDate", "PatientSex", "StudyDate", "StudyTime",
                    "StudyID", "AccessionNumber", "ReferringPhysicianName"):
        if keyword not in ds:
            setattr(ds, keyword, "")
    if "StudyInstanceUID" not in ds:
        ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = seriesUID
    ds.SeriesNumber = SCREEN_SERIES_NUMBER
    ds.InstanceNumber = int(instanceNumber)
    ds.Modality = "OT"
    ds.ConversionType = "WSD"   # workstation
    ds.SeriesDescription = seriesDescription[:64]
    ds.Manufacturer = MANUFACTURER
    ds.SeriesDate = ds.ContentDate = ds.InstanceCreationDate = now.strftime("%Y%m%d")
    ds.SeriesTime = ds.ContentTime = ds.InstanceCreationTime = now.strftime("%H%M%S")
    ds.PatientOrientation = ""
    ds.BurnedInAnnotation = "YES"   # the views may show the patient name and ID
    if comment:
        ds.ImageComments = comment[:10240]
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.PlanarConfiguration = 0
    ds.Rows, ds.Columns = int(rgb.shape[0]), int(rgb.shape[1])
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    data = rgb.tobytes()
    ds.PixelData = data + b"\0" if len(data) % 2 else data   # even length
    return ds


def writeDataset(dataset, path):
    try:
        dataset.save_as(path, enforce_file_format=True)    # pydicom 3
    except TypeError:
        dataset.save_as(path, write_like_original=False)   # pydicom 2


def safeName(text, fallback="case"):
    text = "".join(c if c.isalnum() or c in "-_" else "_" for c in (text or "").strip())
    return text.strip("_")[:40] or fallback


def stamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# --- Reference study and its patient --------------------------------------------------------------------------

def _slicer():
    import slicer
    return slicer


def dicomFiles(node):
    """The DICOM files (in Slicer's DICOM database) a volume or gated sequence was loaded from: its
    DICOM.instanceUIDs, else the instance UIDs of its subject hierarchy item (as the DICOM SPECT Recon extension
    sets them on its reconstructions)."""
    slicer = _slicer()
    if node is None or getattr(slicer, "dicomDatabase", None) is None:
        return []
    candidates = [node]
    if node.IsA("vtkMRMLSequenceNode"):
        try:
            browser = slicer.modules.sequences.logic().GetFirstBrowserNodeForSequenceNode(node)
            proxy = browser.GetProxyNode(node) if browser else None
            if proxy is not None:
                candidates.append(proxy)
        except Exception:
            pass
        if node.GetNumberOfDataNodes():
            candidates.append(node.GetNthDataNode(0))
    shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    for candidate in candidates:
        uids = (candidate.GetAttribute("DICOM.instanceUIDs") or "").split()
        if not uids and shNode is not None:
            item = shNode.GetItemByDataNode(candidate)
            if item:
                uidName = slicer.vtkMRMLSubjectHierarchyConstants.GetDICOMInstanceUIDName()
                uids = (shNode.GetItemUID(item, uidName) or "").split()
        files = []
        for uid in uids:
            try:
                fileName = slicer.dicomDatabase.fileForInstance(uid)
            except Exception:
                fileName = ""
            if fileName and os.path.exists(fileName):
                files.append(fileName)
        if files:
            return files
    return []


def referenceNodes():
    """Volumes / sequences of the exam to file the exports under: the active study first, then the studies of
    the exam (Data roles None left out)."""
    from MyocardialPerfusionImagingLib import session
    from MyocardialPerfusionImagingLib import studies as studiesLib
    nodes = [session.activeStudy()]
    try:
        for study in studiesLib.scanStudies():
            if study.role != studiesLib.EXCLUDED_ROLE:
                nodes += [study.staticNode, study.gatedNode]
    except Exception:
        logging.warning("MPI export: could not list the studies", exc_info=True)
    unique = []
    for node in nodes:
        if node is not None and not studiesLib.isExcluded(node) and node not in unique:
            unique.append(node)
    return unique


def referenceFile():
    """(DICOM file, node) of the first exam study that came from DICOM, or (None, None)."""
    for node in referenceNodes():
        files = dicomFiles(node)
        if files:
            return files[0], node
    return None, None


def studyTags(fileName):
    """Patient and study attributes of a DICOM file (keyword: value, as text)."""
    import pydicom
    dataset = pydicom.dcmread(fileName, stop_before_pixels=True)
    tags = {}
    for keyword in STUDY_TAGS:
        value = dataset.get(keyword)
        if value is not None and str(value) != "":
            tags[keyword] = str(value)
    return tags


def caseTags():
    """Without a DICOM study: the case name and ID as patient, and a study UID kept on the session node (one study
    for every export of the scene)."""
    from pydicom.uid import generate_uid
    from MyocardialPerfusionImagingLib import case as caseLib
    from MyocardialPerfusionImagingLib import session
    name, patientId, _ = caseLib.currentCase()
    tags = {}
    if name:
        family, _, given = name.partition(", ")
        tags["PatientName"] = f"{family}^{given}" if given else name
    if patientId:
        tags["PatientID"] = patientId
    node = session.sessionNode()
    studyUid = node.GetAttribute(EXPORT_STUDY_ATTRIBUTE)
    if not studyUid:
        studyUid = generate_uid()
        node.SetAttribute(EXPORT_STUDY_ATTRIBUTE, studyUid)
    tags["StudyInstanceUID"] = studyUid
    tags["StudyDescription"] = "Myocardial perfusion (Tarvos)"
    return tags


def exportTags():
    """(patient / study tags, reference DICOM file or None, reference node or None)."""
    fileName, node = referenceFile()
    if fileName:
        try:
            return studyTags(fileName), fileName, node
        except Exception:
            logging.warning(f"MPI export: could not read {fileName}", exc_info=True)
    return caseTags(), None, None


def caseLabel():
    from MyocardialPerfusionImagingLib import case as caseLib
    name, patientId, _ = caseLib.currentCase()
    return safeName(patientId or name)


def addToDatabase(path):
    slicer = _slicer()
    try:
        if slicer.dicomDatabase is not None and slicer.dicomDatabase.isOpen:
            slicer.dicomDatabase.insert(path)
            return True
    except Exception as e:
        logging.warning(f"MPI export: could not add {path} to the DICOM database: {e}")
    return False


# --- Screen grabs -----------------------------------------------------------------------------------------------

def _monitor2Windows():
    import qt
    from MyocardialPerfusionImagingLib import views
    titles = (views.GATED_MONITOR2_TITLE, views.PERFUSION_MONITOR2_TITLE)
    return [w for w in qt.QApplication.topLevelWidgets()
            if w.windowTitle in titles and w.visible and not w.inherits("QMainWindow")]


def _renderViews():
    slicer = _slicer()
    layoutManager = slicer.app.layoutManager()
    for index in range(layoutManager.threeDViewCount):
        layoutManager.threeDWidget(index).threeDView().forceRender()
    for name in layoutManager.sliceViewNames():
        layoutManager.sliceWidget(name).sliceView().forceRender()
    slicer.app.processEvents()


def grabScreens():
    """[(caption, QImage)]: the view area of the main window, then each Monitor 2 window shown."""
    import ctk
    slicer = _slicer()
    layoutManager = slicer.app.layoutManager()
    if layoutManager is None:
        raise ExportError("There is no view to capture.")
    _renderViews()
    grabs = [("Views", ctk.ctkWidgetsUtils.grabWidget(layoutManager.viewport()))]
    for window in _monitor2Windows():
        grabs.append((window.windowTitle, ctk.ctkWidgetsUtils.grabWidget(window)))
    return [(caption, image) for caption, image in grabs if image is not None and not image.isNull()]


def imagePixels(image):
    """RGB pixels (rows, columns, 3) of a QImage (written as PPM to a temporary file and read back)."""
    import qt
    slicer = _slicer()
    handle, path = tempfile.mkstemp(suffix=".ppm", dir=slicer.app.temporaryPath)
    os.close(handle)
    try:
        if not image.convertToFormat(qt.QImage.Format_RGB32).save(path, "PPM"):
            raise ExportError("The screenshot could not be written.")
        with open(path, "rb") as f:
            return ppmPixels(f.read())
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _layoutName():
    """Name of the step shown, for the series description."""
    from MyocardialPerfusionImagingLib import session
    key = session.currentStep()
    return next((title for k, _, title, _ in session.STEPS if k == key), "Overview")


# --- Exports ----------------------------------------------------------------------------------------------------

def exportScreen(folder):
    """The screen as DICOM Secondary Capture (one instance per window) in folder. Returns (paths, reference node or
    None, added to the database)."""
    from pydicom.uid import generate_uid
    grabs = grabScreens()
    if not grabs:
        raise ExportError("The views could not be captured.")
    tags, _, reference = exportTags()
    seriesUid = generate_uid()
    description = f"Tarvos {_layoutName()} screen"
    base = f"Tarvos_screen_{caseLabel()}_{stamp()}"
    paths, added = [], True
    for number, (caption, image) in enumerate(grabs, start=1):
        dataset = secondaryCaptureDataset(imagePixels(image), tags, seriesUid, number, description,
                                          f"{caption}: screen capture of 3D Slicer, Tarvos {_layoutName()}")
        path = os.path.join(folder, f"{base}_{number}.dcm" if len(grabs) > 1 else f"{base}.dcm")
        writeDataset(dataset, path)
        paths.append(path)
        added = addToDatabase(path) and added
    return paths, reference, added


def exportScene(folder):
    """The whole scene as a Slicer Data Bundle DICOM file in folder. Returns ([path], reference node or None, added
    to the database)."""
    import DICOMLib
    slicer = _slicer()
    tags, fileName, reference = exportTags()
    tags["SeriesNumber"] = str(SCENE_SERIES_NUMBER)
    tags["Modality"] = "OT"   # not the modality of the studies
    tags["Manufacturer"] = MANUFACTURER
    work = tempfile.mkdtemp("", "TarvosScene", slicer.app.temporaryPath)
    try:
        exporter = DICOMLib.DICOMExportScene(fileName, work)
        exporter.optionalTags = tags
        exporter.seriesDescription = "Tarvos scene"
        if not exporter.export() or not exporter.sdbFile or not os.path.exists(exporter.sdbFile):
            raise ExportError("Slicer could not write the scene bundle (see the log).")
        path = os.path.join(folder, f"Tarvos_scene_{caseLabel()}_{stamp()}.dcm")
        shutil.copyfile(exporter.sdbFile, path)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return [path], reference, addToDatabase(path)


# --- Interactive (toolbar buttons) -----------------------------------------------------------------------------

def _settingsFolder(key):
    import qt
    folder = qt.QSettings().value(key)
    return folder if folder and os.path.isdir(folder) else qt.QDir.homePath()


def _remember(key, folder):
    import qt
    settings = qt.QSettings()
    settings.setValue(key, folder)
    settings.sync()


def chooseDicomFolder():
    """Ask for the folder the DICOM exports are written to (kept in the Slicer settings). Returns it or ""."""
    import qt
    slicer = _slicer()
    folder = qt.QFileDialog.getExistingDirectory(slicer.util.mainWindow(), "Folder for the DICOM exports",
                                                 _settingsFolder(SETTING_DICOM_FOLDER))
    if folder:
        _remember(SETTING_DICOM_FOLDER, folder)
    return folder


def dicomFolder():
    """The folder of the DICOM exports: the one chosen before (asked only the first time, or when it is gone)."""
    import qt
    folder = qt.QSettings().value(SETTING_DICOM_FOLDER)
    return folder if folder and os.path.isdir(folder) else chooseDicomFolder()


def _report(what, paths, reference, added):
    slicer = _slicer()
    lines = [f"{what} saved:"] + paths
    if reference is not None:
        lines.append(f"Filed under the patient and study of '{reference.GetName()}'.")
    else:
        lines.append("No study was loaded from DICOM: filed under the case name and ID, in a study of its own.")
    lines.append("Added to the DICOM database (DICOM browser: send it to a PACS from there)." if added
                 else "Not added to the DICOM database.")
    lines.append("Another folder: Save > Choose DICOM folder...")
    slicer.util.infoDisplay("\n".join(lines), windowTitle="Saved as DICOM")


def _run(title, function):
    """function() with a wait cursor; errors are shown in a message box. Returns its result or None."""
    import qt
    slicer = _slicer()
    slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
    try:
        result = function()
    except ExportError as e:
        slicer.app.restoreOverrideCursor()
        slicer.util.errorDisplay(str(e), windowTitle=title)
        return None
    except Exception as e:
        slicer.app.restoreOverrideCursor()
        logging.exception(f"MPI: {title} failed")
        slicer.util.errorDisplay(f"{title} failed:\n{e}", windowTitle=title)
        return None
    slicer.app.restoreOverrideCursor()
    return result


def _runDicom(what, function):
    folder = dicomFolder()
    if not folder:
        return None
    result = _run(f"Save {what.lower()} as DICOM", lambda: function(folder))
    if result is not None:
        _report(what, *result)
    return result


def saveScreenAsDicom():
    return _runDicom("Screen", exportScreen)


def saveSceneAsDicom():
    return _runDicom("Scene", exportScene)


def _askSaveFile(title, settingKey, fileName, fileFilter):
    import qt
    slicer = _slicer()
    path = qt.QFileDialog.getSaveFileName(slicer.util.mainWindow(), title,
                                          os.path.join(_settingsFolder(settingKey), fileName), fileFilter)
    if path:
        _remember(settingKey, os.path.dirname(path))
    return path


def saveSceneBundle():
    """The whole scene in one .mrb file named after the case. Returns the path or None."""
    slicer = _slicer()
    path = _askSaveFile("Save scene", SETTING_SCENE_FOLDER, f"Tarvos_{caseLabel()}_{stamp()}.mrb",
                        "Medical Reality Bundle (*.mrb)")
    if not path:
        return None
    if not path.lower().endswith(".mrb"):
        path += ".mrb"

    def save():
        if not slicer.util.saveScene(path):
            raise ExportError(f"Slicer could not save the scene to\n{path}\n(see the log).")
        return path
    if _run("Save scene", save) is None:
        return None
    slicer.util.showStatusMessage(f"Scene saved: {path}", 5000)
    return path


def openSaveDialog():
    """Slicer's own Save dialog (every node as its own file, chosen formats)."""
    _slicer().app.ioManager().openSaveDataDialog()


def saveScreenshot():
    """The views (and the Monitor 2 window) as PNG files. Returns the paths or None."""
    slicer = _slicer()
    path = _askSaveFile("Save screenshot", SETTING_SCREENSHOT_FOLDER, f"Tarvos_{caseLabel()}_{stamp()}.png",
                        "PNG image (*.png)")
    if not path:
        return None
    if not path.lower().endswith(".png"):
        path += ".png"

    def save():
        grabs = grabScreens()
        if not grabs:
            raise ExportError("The views could not be captured.")
        root = path[:-4]
        paths = []
        for number, (_, image) in enumerate(grabs, start=1):
            target = path if number == 1 else f"{root}_{number}.png"
            if not image.save(target, "PNG"):
                raise ExportError(f"Could not write {target}.")
            paths.append(target)
        return paths
    paths = _run("Save screenshot", save)
    if paths:
        slicer.util.showStatusMessage("Screenshot saved: " + ", ".join(paths), 5000)
    return paths
