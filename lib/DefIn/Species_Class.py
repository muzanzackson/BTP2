#!/usr/bin/env python

#===========================================================================================================================================================
# Example of clades of Gadiformes class

# Gadinae
family1 = ['Arctogadus_glacialis', 'Micromesistius_poutassou', 'Boreogadus_saida', 'Pollachius_pollachius', 'Pollachius_virens', 'Gadus_morhua_kildinensis', 'Theragra_chalcogramma_pantophysin', 'Gadus_ogac', 'Theragra_finnmarchica', 'Melanogrammus_aeglefinus']
# Lotinae
family2 = ['Lota_lota']
# Marlucciidae
family3 = ['Merluccius_merluccius']
# Macrouridae
family4 = ['Cetonurus_globiceps', 'Coelorinchus_kishinouyei', 'Ventrifossa_garmani']
# Bathydadinae
family5 = ['Bathygadus_antrodes']
# Bregmacerotidae
family6 = ['Bregmaceros_nectabanus']
# Trachyrincinae
family7 = ['Trachyrincus_murrayi']
# Macrouridinae
family8 = ['Squalogadus_modificatus']
# Outgroup
family9 = ['Sardinops_melanostictus']

# Forming clades based on the knowledge
# Gadinae is monophyly
clade_1 = ['Arctogadus_glacialis', 'Micromesistius_poutassou', 'Boreogadus_saida', 'Pollachius_pollachius', 'Pollachius_virens', 'Gadus_morhua_kildinensis', 'Theragra_chalcogramma_pantophysin', 'Gadus_ogac', 'Theragra_finnmarchica', 'Melanogrammus_aeglefinus']
# Gadinae and Lotinae are sisters
clade_2 = ['Arctogadus_glacialis', 'Micromesistius_poutassou', 'Boreogadus_saida', 'Pollachius_pollachius', 'Pollachius_virens', 'Gadus_morhua_kildinensis', 'Theragra_chalcogramma_pantophysin', 'Gadus_ogac', 'Theragra_finnmarchica', 'Melanogrammus_aeglefinus','Lota_lota']
# Marlucciidae is sister of both Gadinae and Lotinae
clade_3 = ['Arctogadus_glacialis', 'Micromesistius_poutassou', 'Boreogadus_saida', 'Pollachius_pollachius', 'Pollachius_virens', 'Gadus_morhua_kildinensis', 'Theragra_chalcogramma_pantophysin', 'Gadus_ogac', 'Theragra_finnmarchica', 'Melanogrammus_aeglefinus','Lota_lota','Merluccius_merluccius']
# Macrouridae forms monophyletic clade
clade_4 = ['Cetonurus_globiceps', 'Coelorinchus_kishinouyei', 'Ventrifossa_garmani']
# Trachyrincinae and Macrouridinae form monophyletic clade
clade_5 = ['Trachyrincus_murrayi', 'Squalogadus_modificatus']
#  Gadinae, Lotinae, Marlucciidae, Macrouridae, Trachyrincinae, and Macrouridinae form monophyletic clade
clade_6 = ['Arctogadus_glacialis', 'Micromesistius_poutassou', 'Boreogadus_saida', 'Pollachius_pollachius', 'Pollachius_virens', 'Gadus_morhua_kildinensis', 'Theragra_chalcogramma_pantophysin', 'Gadus_ogac', 'Theragra_finnmarchica', 'Melanogrammus_aeglefinus','Lota_lota','Merluccius_merluccius', 'Cetonurus_globiceps', 'Coelorinchus_kishinouyei', 'Ventrifossa_garmani', 'Trachyrincus_murrayi', 'Squalogadus_modificatus']
# Bregmacerotidae placed at the most outer part
clade_7 = ['Arctogadus_glacialis', 'Micromesistius_poutassou', 'Boreogadus_saida', 'Pollachius_pollachius', 'Pollachius_virens', 'Gadus_morhua_kildinensis', 'Theragra_chalcogramma_pantophysin', 'Gadus_ogac', 'Theragra_finnmarchica', 'Melanogrammus_aeglefinus','Lota_lota','Merluccius_merluccius', 'Cetonurus_globiceps', 'Coelorinchus_kishinouyei', 'Ventrifossa_garmani', 'Trachyrincus_murrayi', 'Squalogadus_modificatus','Bregmaceros_nectabanus']

all_clades = [clade_1, clade_2, clade_3, clade_4, clade_5, clade_6, clade_7]

