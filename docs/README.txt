-------------------------------------------------------------------------------------
- A CNN-Transformer Model for Extracting Hand Gesture Inforamtion from sEMG Signals -
-------------------------------------------------------------------------------------

Place sEMG-CNNTransformer folder in the 'projects' folder of CC directory.

*IMPORTANT*: In order to successfully run the code, two steps must be taken:

1. In the sEMG-CNNTransformer folder, create a new virtual environment 'newENV' (virtualenv --no-download newENV); source this environment and 
install all the requirements in 'requirements.txt'.

2. Due to disk quota issues, the zipped Ninapro DB2 files could not be added from http://ninapro.hevs.ch/data2; download 
the zipped data for each subject from the website and place in data/zipped folder. Extraction can be done with the 
preprocessing code (see src/data_preprocess/preprocess_main.py).

Preprocessing can be run by calling 'run_preprocess.bash'; see 'preprocess_main.py' for details about additional arguments that can be added to the file call in bash script.

Training can be run by calling 'run_training.bash'; see 'training_main.py' for details about additional arguments that can be added to the file call in bash script.

Model compression can be run by calling 'run_compression.bash'; see 'compression_main.py' for details about additional arguments that can be added to the file call in bash script.

Any bash script can be run using the command 'sbatch TODO.bash'. Run from projects/sEMG-CNNTransformer.

------------------
- FILE STRUCTURE -
------------------

A brief overview of the  repository structure is as follows:

sEMG-CNNTransformer
|- data                       (Location of all training and testing data)
| |- processed                (Processed data, see section NAMING CONVENTIONS for file name explanation)
| |- raw                      (Unzipped data, before processing)
| |- zipped                   (Zipped data collected from http://ninapro.hevs.ch/data2**)
|- docs                       (Documentation)
|- images                     (Output folder for images, e.g. confusion matrices)
| |- active                   (Location for images generated after 05/04/2023)
| |- archive                  (Location for images generated before 05/04/2023)
|- models                     (Output folder for trained models)
| |- active                   (Location for models generated after 05/04/2023)
| |- archive                  (Location for models generated before 05/04/2023)
| |- checkpoints              (Location for model checkpoints)
|- src                        (Main code location)
| |- data_preprocess          (Data preprocessing tools)
| | |- preprocess_utils.py
| | |- preprocess_main.py
| |- model_compression        (Model compression / quantization tools)
| | |- compression_utils.py
| | |- compression_main.py
| |- model_training           (Model training tools)
| | |- training_utils.py
| | |- training_main.py
|- logs                       (Location for output logs from Slurm logs)
|- newENV                     (Environment for Slurm runs)

----------------------
- NAMING CONVENTIONS -
----------------------

- data/processed: {SUBJECT}_{NORMALIZATION}_{FREQ}_{FILTER}_{RMSCONT}_{RMSNONCONT}_{train/test}_{x/y}.npy

SUBJECT: Subject number.
NORMALIZATION: Normalized using z-score (bool).
FREQ: Downsampled frequency of data, e.g. '1000'.
FILTER: Type of Butterworth filter employed (order, type, cutoff), e.g. 'none', '1lowpass10'. 
RMSCONT: If continuous RMS processing is used (bool).
RMSNONCONT: If noncontinuous RMS processing is used (bool).

- models: {RUNID}_{SUBJECT}_{MODEL}_{EXTRADESC}_{ACCURACY}.h5

RUNID: Slurm ID run.
MODEL: Type of model, e.g. 'CNNTransformer'.
EXTRADESC: Description tag about the model architecutre, e.g. '2attnheads'.
ACCURACY: Validation accuracy of model in decimal form.

- images: {TYPE}_{RUNID}_{SUBJECT}_{MODEL}_{EXTRADESC}_{ACCURACY}.png

TYPE: The type of images, e.g. 'confusion_matrix'.

*NOTE: Models and images in the archive/ folders may not all follow these conventions.
