## About DefIn

__`DefIn`__ is a python based program for computing the Deformity Index of a phylogenetic tree based on either partial or complete knowledge of the relationships among the species. 


## Required packages

The DefIn is developed on python 2.7 and Linux based system. So this program is only compatible for python 2.7. The other prerequisite is,

- Dendropy v4.4.0


## Create the environment

For executing the DefIn we recommend to create a separate environment in anaconda. The image of the required environment (`.yml`) is given in this package. First create the environment and activate it by executing the following commands,

`conda env create -f dienv.yml`

`conda activate diEnv`

We recommend to verify the installed package list in the `diEnv` by executing `conda env list`.


## The DefIn options

DefIn can only be executed through the command prompt. It is necessary to make all the python file executable before running the program (use `chmod +x *.py`). The command to execute the DefIn from the command prompt is as `./DefIn.py` with the command line arguments. The options are as follows,

- `-i`	Mention the file that contains the target tree (in NEWICK format). Please refer to the _`Sample_Input/TargetTree.nwk`_ for the detail format of the file.

- `-t`	This is a boolean option. If this option is provided then the program will consider the reference tree mentioned by the option `-r`, else the program will consider the list of clades mentioned by the option `-c`. For further detail please refer to option `-c` and `-r`.

- `-c`	Mention the file that contains the list of clades. The clade is represented as the list of species separated by a comma. Each line contains a single clade. Please refer to the _`Sample_Input/Clades.txt`_ for the detail format of the file.

- `-r`	Mention the file that contains the reference tree (in NEWICK format). The reference tree can be both resolved (binary) or unresolved. Please refer to the _`Sample_Input/RefTree.nwk`_ for the detail format of the file.


## Execution of DefIn

`./DefIn.py -i [target tree] [ -c | [ -r [reference tree] -t ] ]`

- Minimal command:

`./DefIn.py -i [target_tree]`

In the minimal command, the script considers `Species_Class.py` placed in the same directory. Please refer to the `Species_Class.py` for the detail format of the file. 

The DefIn can execute in two different modes:

- Using list of clades: In this command, the script considers the file contains a list of clades. The list of clades can be provided in two ways, 

	(a) (Default) providing the list of clades in `Species_Class.py` placed in the same directory.

	(b) providing the list of clades as a text file. For this use the option `-c` to mention the text file.

- Using reference tree: In this command, the script considers the file contains a reference tree to compute the Deformity Index. For this the option `-t` need to put and the reference tree file should be mentioned using `-r` option.


** A sample command file is provided with this package. The shell script file name `Sample_Script.sh`, contains the sample code and can be executed by changing the variables accordingly. 

