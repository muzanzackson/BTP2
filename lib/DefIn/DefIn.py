#!/usr/bin/env python

from Header import *
import Species_Class
from Species_Class import *
import Parsing
from Parsing import *


##---------------------------------------------------------------------------------
''' this function is useful to parse various options for input data processing '''
##---------------------------------------------------------------------------------
def parse_options():  
	parser = OptionParser()

	parser.add_option("-i", "--INPTREE", \
		type="string", \
		action="store", \
		dest="INP_TREE", \
		default="", \
		help="path of the target tree")	
			
	parser.add_option("-t", "--ISREFTREE", \
		action="store_true", \
		dest="IS_REFTREE", \
		default=False, \
		help="If TRUE, then it will consider tree, ELSE, it will consider clade")	
	
	parser.add_option("-c", "--CLADEFILE", \
		type="string", \
		action="store", \
		dest="CLADE_FILE", \
		default="Species_Class.py", \
		help="path of the file stores the list of clades")		
	
	parser.add_option("-r", "--REFTREE", \
		type="string", \
		action="store", \
		dest="REF_TREE", \
		default="", \
		help="path of the reference tree")	
	
	opts, args = parser.parse_args()
	return opts, args  



#-------------------------------------------------------------------------
# The transer_in to be computed for the case where the species placed 
# outside of the clade and transfer into the clade
#-------------------------------------------------------------------------
def Transer_in(Inp_Tree, intr_node, NODE_LEVEL, correct_ingrps, wrong_outgrps, missed_ingrps, TREE_LEAVES, all_leaves_label):	
	trans_in = 0
	for taxon_label in missed_ingrps:
		if taxon_label in all_leaves_label:
			if len(correct_ingrps) == 0:
				leaves =list(set(wrong_outgrps)|{taxon_label})
			else:
				leaves =list(set(correct_ingrps)|{taxon_label})
						
			mrca = Inp_Tree.mrca(taxon_labels=leaves)
			leaf_nodes = mrca.leaf_nodes()
			leaf_labels = [ln.taxon.label for ln in leaf_nodes]
			
			mrca_level = NODE_LEVEL[mrca]
			miss_node = Inp_Tree.find_node_with_taxon_label(taxon_label)
			parent_miss_node = miss_node.parent_node
			parent_level = NODE_LEVEL[parent_miss_node]
			curr_node_level = NODE_LEVEL[intr_node]
			
			trans_in += (parent_level - mrca_level) + (curr_node_level - mrca_level)
	
	return trans_in, len(missed_ingrps)




#-------------------------------------------------------------------------
# The transer_out to be computed for the case where the outgroup 
# species placed in a clade and transfer outside the clade
#-------------------------------------------------------------------------
def Transer_out(Inp_Tree, intr_node, NODE_LEVEL, wrong_outgrps, TREE_LEAVES):
	trans_out = 0
	for taxon_label in wrong_outgrps:
		wrong_node = Inp_Tree.find_node_with_taxon_label(taxon_label)
		parent_wrong_node = wrong_node.parent_node
		parent_level = NODE_LEVEL[parent_wrong_node]
		curr_node_level = NODE_LEVEL[intr_node]
		
		trans_out += parent_level - curr_node_level + 1
	
	return trans_out, len(wrong_outgrps)
	
	

#-------------------------------------------------------------------------
# For a given clade, the deformity scores are computed for each of the 
# internal node of the tree 
#-------------------------------------------------------------------------
def Deformity_Score_for_Clade(Inp_Tree, clade, NODE_LEVEL, internal_nodes, TREE_LEAVES, all_leaves_label):
	deformity_score = []
	common_species = list(set(clade) & set(all_leaves_label))
	
	if len(common_species) > 1:
		clade = common_species
		for intr_node in internal_nodes:		
			leaf_nodes = intr_node.leaf_nodes()
			leaf_labels = [ln.taxon.label for ln in leaf_nodes]
			
			missed_ingrps = list(set(clade)-set(leaf_labels))
			correct_ingrps = list(set(clade)-set(missed_ingrps))
			wrong_outgrps = list(set(leaf_labels)-set(clade))
			
			trans_in, m = Transer_in(Inp_Tree, intr_node, NODE_LEVEL, correct_ingrps, wrong_outgrps, missed_ingrps, TREE_LEAVES, all_leaves_label)				
			trans_out, n = Transer_out(Inp_Tree, intr_node, NODE_LEVEL, wrong_outgrps, TREE_LEAVES)
			
			ds_clade = (trans_in+trans_out) * len(clade) * 1.0 / len(leaf_nodes)		
			deformity_score.append(ds_clade)
	
	if len(deformity_score) == 0:
		min_deform = 0
	else:
		min_deform = min(deformity_score)

	return min_deform



def main():
	opts, args = parse_options()
	
	INP_TREE = opts.INP_TREE	
	IS_REFTREE = opts.IS_REFTREE
	REF_TREE = opts.REF_TREE
	CLADE_FILE = opts.CLADE_FILE	
	
	deformity_index = []
	
	if IS_REFTREE is True:
		# fp_reftree = open(REF_TREE,'r')
		# tree = fp_reftree.readline()
		# Read ref tree
		ref_tree = Tree.get(path=REF_TREE, schema='newick', preserve_underscores=True)
		#ref_tree = dendropy.Tree.get_from_string(tree,'newick',preserve_underscores=True)	
		# no_of_species = len(ref_tree.leaf_nodes())
		
		# Extract all clades
		internal_nodes = ref_tree.internal_nodes()
		ref_clade = []
		for intr_node in internal_nodes:
			leaf_nodes = intr_node.leaf_nodes()
			leaf_labels = [ln.taxon.label for ln in leaf_nodes]
			ref_clade.append(leaf_labels)		
		all_clades = ref_clade			
		selected_clades = all_clades
	else:
		# Read the list of clades from Species_Class.py
		if CLADE_FILE == 'Species_Class.py':
			selected_clades = Species_Class.all_clades	
		else:
			selected_clades = Parsing.Parse_Clade(CLADE_FILE)
			if selected_clades == -1:
				sys.exit()
		
	no_of_class = len(selected_clades)
	print ('Number of clades: ', no_of_class)
	
	# fp_tree = open(INP_TREE,'r')
	# line = fp_tree.readline()
	# target tree read
	tree = Tree.get(path=INP_TREE, schema='newick', preserve_underscores=True)
	# tree = dendropy.Tree.get_from_string(line,'newick',preserve_underscores=True)
	#print tree.as_ascii_plot()
			
	TREE_LEAVES = tree.leaf_nodes()
	all_leaves_label = [ln.taxon.label for ln in TREE_LEAVES]
	
	deformity_score_clade = []
	
	nodes = tree.preorder_node_iter()
	NODE_LEVEL = dict()	
	for nd in nodes:
		NODE_LEVEL.update({nd : nd.level()})
				
	internal_nodes = tree.internal_nodes()
			
	for fm in range(no_of_class):
		#this condition check whether the fm is root node. 
		#If it is root node then it will not compute deformity at that clade (Trivial).
		if bool(set(all_leaves_label)-set(selected_clades[fm])):		
			ds_clade = Deformity_Score_for_Clade(tree,selected_clades[fm],NODE_LEVEL, internal_nodes, TREE_LEAVES, all_leaves_label)
			deformity_score_clade.append(ds_clade)
		else:
			deformity_score_clade.append(0)
	# print deformity_score_clade
			
	d_index = sum(deformity_score_clade) * 1.0 / len(deformity_score_clade)
		
	print( 'DEFORMITY INDEX: ', d_index)
#---------------------------------------------------------------------------------------

if __name__=='__main__':
	#print clade
	main()
