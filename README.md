# PING Analyzer

## Plan


This utility will help analyze the data that we recieve from PYXIS. We will be able to integrate with ROOT and read in data files. Using various analysis methods, we will then calculate important metrics such as beam position. Finally, we plan to add support for a GUI that visualizes the beam and detector, as well as support for non-PYXIS detectors.

# Detector
<p align="center">
  <img src="https://github.com/hydrol0x/PYXIS-python/assets/34951139/907a8bf0-0b24-4619-b9f6-0559747f62ff" alt="Visual depiction of `Detector` class"/>
</p>

The `Detector` abstract class defines a blueprint for how any detector (that will be) supported by the PING analyzer program works. It must have the specified functions/attributes in order to be supported, but the details of implementation (for example how beam position is calculated) is left as a 'black box' to the class. Position/momentum (will be implemented) should return the beam position with time in [specific format will be specified once implemented]. 

The `generate()` method is used to generate plausible but mock data for the particular detector. The `file_name()` method returns the default file name that the generated data will be written to, i.e `./[file_name].root.` 

## Old
Check `old/` for the old test data generator that we wrote. We are now working on a new version that will be used with PYXIS and other detectors when they are built.
