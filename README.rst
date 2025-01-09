==========================
Pseudo-MRI engine
==========================

Overview
========

Pseudo-MRI engine utilizes the head shape digitization points, generally acquired during MEG/EEG acquisition, for accurately warping a given MRI template to the best fit to the subject's head. 
It yields a set of robustly warped MRI and its derivative surfaces, referred to as pseudo-MRI. Although the dense and evenly sampled scalp digitization points are recommended, the software can efficiently generate pseudo-MRI even with as few as 25 points (nearly 1 point per 25 cm2).
Pseudo-MRI-engine is a pure Python package, and depends on several other Python packages; it requires Python version 3.7 or higher.

| **GitHub Repository**: https://github.com/neurosignal/pseudo-MRI-engine


Features
--------
• It can generate a subject-specific template MRI, i.e., pseudo-MRI, based on the head shape. 
• Such a pseudo-MRI can readily be used with severals oftware packages for MEG/EEG source imaging.
• Easy-to-use interface for quick integration into workflows.
• Open-source and extendable.


Quick Start
===========

Installation
------------

Download the software:

.. code-block:: bash

    git clone https://github.com/neurosignal/pseudo-MRI-engine.git

Change the directory:

.. code-block:: bash

    cd pseudo-MRI-engine

Set up a Python environment, for example as:

.. code-block:: bash

    conda activate <your_environment>

Run to check and install all dependencies for the pseudo-MRI engine:

.. code-block:: bash

    pip install -r requirements.txt

Check the installation: 

.. code-block:: bash

    python pseudoMRI_engine.py --help


Preparation of a template MRI
-----------------------------

Segment a template MRI, for example MNE152, using recon-all routine of FreeSurfer (or FastSurfer). Also compute head model and scalp surfaces using MNE-Python routines. Furthermore, the fiducial points for the template MRI can be identified using MNE-Python coregistration module and saved as FIFF file with an appropriate pattern from CrusHelix, Targus, and ITnotch.


Usage
-----

Check installation and input arguments:

.. code-block:: bash

    python pseudo-MRI-engine.py --help

Run as
------
.. code-block:: bash

    python pseudoMRI_engine.py --pseudo_MRI_name <subject ID> --pseudo_MRI_dir <pseudo-MRI folder> --headshape <headshape file> --template_MRI_name <name of template MRI folder> --template_MRI_dir <the parent directory of the template MRI folder> --fiducial_file <fiducial file of the template MRI> --preauri_loc <the position of the LPA/RPA considered during the head digitization> --nmax_Ctrl <maximum number of the control points to compute warping> --dense_hsp <set this flag to force densifying the digitization points if too sparse> --open_report <set this flag to open the HTML report file in the end>

Example
-------

The subdirectory *data* includes sample data for ice-breaking and test run to start with *pseudo-MRI-engine*. 
The *templates* under *data* includes a template MRI (ICBM2009cNolinAsym; Fonov et al., 2011), prepared using FreeSurfer v7.4.1 (Fischl, 2012). The *headshapes* has a MEG file *test_case.fif* recorded from a 35-years-old healthy adult using MEGIN's MEG system. This file also holds the digitization data defining the subject's head shape.

.. code-block:: bash

    python pseudoMRI_engine.py --pseudo_MRI_name ICBM2009cNolinAsym_test_case \
    --pseudo_MRI_dir data/templates/ --headshape data/headshapes/test_case.fif \
    --template_MRI_name ICBM2009cNolinAsym --template_MRI_dir data/templates/ \
    --preauri_loc CrusHelix --nmax_Ctrl 200 --which_mris T1.mgz,brain.mgz --open_report

Requirements
============

- glob2==0.7

- ipython==8.12.3

- IsoScore==2.0.1

- matplotlib==3.7.2

- mayavi==4.8.1

- mne==1.9.0

- nibabel==4.0.2

- nilearn==0.9.2

- numpy==1.24.4

- scipy==1.10.1

- Other dependencies (see `requirements.txt`)


Contributing
============

Contributions are welcome! If you have suggestions or find bugs, please open an issue or submit a pull request.





