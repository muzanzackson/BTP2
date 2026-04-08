#!/usr/bin/env python


def Parse_Clade(inp_filename):
	list_of_clades = []
	with open(inp_filename,'r') as fp:
		while True:
			line = fp.readline()
			if line:				
				words = line.strip().split(',')
				for w in range(len(words)):
					wd = words[w]
					if len(wd.strip()) == 0:
						print ('Error: present of blank element(s) in the list of clades.')
						return(-1)
					else:
						words[w] = wd.strip()
					
				list_of_clades.append(words)
			else:
				break
	
	return list_of_clades
		

##---------------------------------------------------------------------------------------

#if __name__=='__main__':
	#Parse_Clade('Sample_Input/Clades.txt')
