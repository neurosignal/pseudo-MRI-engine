# pseudo-MRI-engine
Pseudo-MRI engine for MRI-free electromagnetic source imaging.
It utilizes the head shape digitization points, generally acquired during MEG/EEG acquisition, for warping a given MRI template to the best fit to the subject's head. It yields a set of robustly warped MRI and its derivative surfaces, referred to as pseudo-MRI. Although the dense and evenly sampled scalp digitization points are recommended, the software can efficiently generate pseudo-MRI even with as few as 25 points (nearly 1 point per 25 cm2).

**Installation**

pseudo-MRI-engine is a pure Python package, and depends on several other Python packages. pseudo-MRI-engine requires Python version 3.7 or higher.

**Using source code**

wget https://github.com/neurosignal/pseudo-MRI-engine.git

cd pseudo-MRI-engine

python setup.py install

**How to use**

conda activate <your_python_environment_where_pseudo-MRI-engine_is_installed>

**In terminal**

python pseudo-MRI-engine.py --help

python pseudo-MRI-engine.py --digfile <file_with_digitization_data> --template 


**Dependencies**





