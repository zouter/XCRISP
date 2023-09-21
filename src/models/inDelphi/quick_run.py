import sys

sys.path.append('/Users/colm/repos/inDelphi-model')
import inDelphi

inDelphi.init_model(celltype = 'mESC')
seq = "GTAAAAGTTAAAATATCTTTAACCTAAAACCGGGTTCAGCGACTTATTTAGTCCA"
cutsite = 27

# GTAAAAGTTAAAATATCTTTAACCTAA|AACCGGGTTCAGCGACTTATTTAGTCCA


inDelphi.predict(seq, cutsite)
pred_df, stats = inDelphi.predict(seq, cutsite)
pred_df = inDelphi.add_genotype_column(pred_df, stats)

print("Done.")