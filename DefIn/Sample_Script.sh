#-------------------------------------------------------------------------------------------
# Sample script
#-------------------------------------------------------------------------------------------


# List of input files
TARGET_TREE='Sample_Input/TargetTree.nwk'
LIST_OF_CLADES='Sample_Input/Clades.txt'
REFERENCE_TREE='Sample_Input/RefTree.nwk'

# Minimal command to execute
# In the minimal command, the script considers 'Species_Class.py'
execstr='./DeformityIndex.py -i '$TARGET_TREE
echo $execstr
$execstr

# Using the list of clade file
# In this command, the script considers the file contains list of clades
execstr='./DeformityIndex.py -i '$TARGET_TREE' -c '$LIST_OF_CLADES
echo $execstr
$execstr

# Using the reference tree
# In this command, the script considers the reference tree
execstr='./DeformityIndex.py -i '$TARGET_TREE' -r '$REFERENCE_TREE' -t'
echo $execstr
$execstr
