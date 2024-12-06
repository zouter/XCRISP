import io, csv, os, sys
import pandas as pd
import pickle as pkl
import Bio.Seq
from src.data.data_loader import get_details_from_fasta
from tqdm import tqdm

NTS = ['A','T','G','C']

def feature_DelSize(indel_details ):
    features, feature_labels = [],[]
    uncut_seq, cut_site, left, right, ins_seq = indel_details
    dsize = right-left-1
    feature_labels.append('Any Deletion'); features.append( len(ins_seq) == 0 )
    feature_labels.append('D1'); features.append( len(ins_seq) == 0 and dsize == 1)
    feature_labels.append('D2-3'); features.append( len(ins_seq) == 0 and dsize in range(2,4))
    feature_labels.append('D4-7'); features.append( len(ins_seq) == 0 and dsize in range(4,8))
    feature_labels.append('D8-12'); features.append( len(ins_seq) == 0 and dsize in range(7,13))
    feature_labels.append('D>12'); features.append( len(ins_seq) == 0 and dsize > 12)
    return features, feature_labels

def feature_InsSize(indel_details):
    features, feature_labels = [],[]
    uncut_seq, cut_site, left, right, ins_seq = indel_details
    feature_labels.append('Any Insertion'); features.append( len(ins_seq) > 0 )
    feature_labels.append('I1'); features.append( len(ins_seq) == 1 )
    feature_labels.append('I2'); features.append( len(ins_seq) == 2 )
    return features, feature_labels

def feature_DelLoc(indel_details ):
    features, feature_labels = [],[]
    uncut_seq, cut_site, left, right, ins_seq = indel_details
    for lmin, lmax in [(-1,-1),(-2,-2),(-3,-3),(-4,-6),(-7,-10),(-11,-15),(-16,-30)]:
        feature_labels.append('DL%d-%d' % (lmin,lmax)); features.append( (left - cut_site) in range(lmin,lmax+1) )
    feature_labels.append('DL<-30'); features.append( (left - cut_site)< -30 )
    feature_labels.append('DL>=0'); features.append( (left - cut_site) >= 0 )
    for rmin, rmax in [(0,0),(1,1),(2,2),(3,5),(6,9),(10,14),(15,29)]:
        feature_labels.append('DR%d-%d' % (rmin,rmax)); features.append( (right - cut_site) in range(rmin,rmax+1) )
    feature_labels.append('DR<0'); features.append( (right - cut_site) < 0 )
    feature_labels.append('DR=>30'); features.append( (right - cut_site) > 30 )
    if len(ins_seq) > 0:    #Apply these features to deletions only
        features = [0 for x in features]
    return features, feature_labels

def feature_InsSeq(indel_details ):
    features, feature_labels = [],[]
    uncut_seq, cut_site, left, right, ins_seq = indel_details
    for nt in NTS: 
        feature_labels.append('I1_%s' % (nt)); features.append(ins_seq==nt)
        for nt2 in NTS:
            feature_labels.append('I2_%s%s' % (nt, nt2)); features.append(ins_seq==(nt+nt2))
    return features, feature_labels

def feature_InsLoc(indel_details ):
    features, feature_labels = [],[]
    uncut_seq, cut_site, left, right, ins_seq = indel_details
    for lmin, lmax in [(-1,-1),(-2,-2),(-3,-3)]:
        feature_labels.append('IL%d-%d' % (lmin,lmax)); features.append( (left - cut_site) in range(lmin,lmax+1) )
    feature_labels.append('IL<-3'); features.append( (left - cut_site) < -3 )
    feature_labels.append('IL>=0'); features.append( (left - cut_site) >= 0 )
    if len(ins_seq) == 0:    #Apply these features to insertions only
        features = [0 for x in features]
    return features, feature_labels

def feature_LocalCutSiteSequence(indel_details, lims=(-5, 4) ):
    features, feature_labels = [],[]
    uncut_seq, cut_site, left, right, ins_seq = indel_details
    for offset in range(lims[0],lims[1]):
        for nt in NTS:
            feature_labels.append('CS%d_NT=%s' % (offset, nt)); features.append(uncut_seq[cut_site+offset]==nt)
    return features, feature_labels 

def feature_LocalCutSiteSeqMatches(indel_details, lims=(-3, 2) ):
    features, feature_labels = [],[]
    uncut_seq, cut_site, left, right, ins_seq = indel_details
    for offset1 in range(lims[0],lims[1]):
        for offset2 in range(lims[0],offset1):
            for nt in NTS:
                feature_labels.append('M_CS%d_%d_NT=%s' % (offset1, offset2, nt))
                features.append((uncut_seq[cut_site+offset1]==uncut_seq[cut_site+offset2]) and (uncut_seq[cut_site+offset1]==nt))
    return features, feature_labels 

def feature_LocalRelativeSequence(indel_details, lims=(-3, 3) ):
    features, feature_labels = [],[]
    uncut_seq, cut_site, left, right, ins_seq = indel_details
    for offset in range(lims[0],lims[1]):
        for nt in NTS:
            feature_labels.append('L%d_NT=%s' % (offset, nt)); features.append(uncut_seq[left+1+offset]==nt if left+offset+1>=0 else 0)
            feature_labels.append('R%d_NT=%s' % (offset, nt)); features.append(uncut_seq[right+offset]==nt if right+offset<len(uncut_seq) else 0)
    if len(ins_seq) > 0:    #Apply these features to deletions only
        features = [0 for x in features]
    return features, feature_labels

def features_SeqMatches(indel_details, lims=(-3, 3) ):
    features, feature_labels = [],[]
    uncut_seq, cut_site, left, right, ins_seq = indel_details
    for loffset in range(lims[0],lims[1]):
        for roffset in range(lims[0],lims[1]):
            feature_labels.append('M_L%d_R%d' % (loffset, roffset)) 
            feature_labels.append('X_L%d_R%d' % (loffset, roffset))
            if left+loffset > 0 and right+roffset < len(uncut_seq):
                features.append(uncut_seq[left+loffset+1]==uncut_seq[right+roffset])
                features.append(uncut_seq[left+loffset+1]!=uncut_seq[right+roffset])
            else: features.append(0); features.append(0)
    if len(ins_seq) > 0:    #Apply these features to deletions only
        features = [0 for x in features]
    return features, feature_labels

def features_InsSeqMatches(indel_details ):
    features, feature_labels = [],[]
    uncut_seq, cut_site, left, right, ins_seq = indel_details
    for offset in range(-5, 5):
        feature_labels.append('I1M_O%d' % (offset)); features.append(uncut_seq[cut_site+offset]==ins_seq)
        feature_labels.append('I2M_O%d_N1' % (offset)); features.append(uncut_seq[cut_site+offset]==ins_seq[0])
        feature_labels.append('I2M_O%d_N2' % (offset)); features.append(uncut_seq[cut_site+offset]==ins_seq[1])
        feature_labels.append('I1X_O%d' % (offset)); features.append(uncut_seq[cut_site+offset]!=ins_seq)
        feature_labels.append('I2X_O%d_N1' % (offset)); features.append(uncut_seq[cut_site+offset]!=ins_seq[0])
        feature_labels.append('I2X_O%d_N2' % (offset)); features.append(uncut_seq[cut_site+offset]!=ins_seq[1])
    if len(ins_seq) == 0:    #Apply these features to insertions only
        features = [0 for x in features]
    return features, feature_labels

def feature_I1or2Rpt(indel_details):
    features, feature_labels = [],[]
    uncut_seq, cut_site, left, right, ins_seq = indel_details
    feature_labels.append('I1Rpt'); features.append(ins_seq == uncut_seq[cut_site-1] and (left-cut_site)==-1)
    feature_labels.append('I1NonRpt'); features.append(len(ins_seq) == 1 and ins_seq != uncut_seq[cut_site-1] and (left-cut_site)==-1)
    feature_labels.append('I2Rpt'); features.append(ins_seq == (uncut_seq[cut_site-1]*2) and (left-cut_site)==-1)
    feature_labels.append('I2NonRpt'); features.append(len(ins_seq) == 2 and ins_seq != (uncut_seq[cut_site-1]*2) and (left-cut_site)==-1)
    if len(ins_seq) == 0:    #Apply these features to insertions only
        features = [0 for x in features]
    return features, feature_labels

def hasLeftMH(left,right, uncut_seq, mh_len, mismatch=0):
    left_seq = uncut_seq[left-mh_len:left+1]
    right_seq = uncut_seq[right-mh_len-1:right]
    if len(left_seq[1:]) != len(right_seq[1:]) or len(right_seq[1:])==0: return 0
    if left_seq[0] == right_seq[0]: return 0  #(if part of a longer microhomology)
    if left_seq[-1] != right_seq[-1]: return 0  #(last letter must match)
    if left_seq[1] != right_seq[1]: return 0  #(first letter must match)
    return sum([x==y for (x,y) in zip(left_seq[1:],right_seq[1:])]) == (mh_len-mismatch)

def hasRightMH(left,right, uncut_seq, mh_len, mismatch=0):
    left_seq = uncut_seq[left+1:left+mh_len+2]
    right_seq = uncut_seq[right:right+mh_len+1]
    if len(left_seq[:-1]) != len(right_seq[:-1]) or len(left_seq[:-1])==0: return 0
    if left_seq[-1] == right_seq[-1]: return 0  #(if part of a longer microhomology)
    if left_seq[-2] != right_seq[-2]: return 0  #(last letter must match)
    if left_seq[0] != right_seq[0]: return 0  #(first letter must match)
    return sum([x==y for (x,y) in zip(left_seq[:-1],right_seq[:-1])]) == (mh_len-mismatch)

def feature_microhomology(indel_details ):
    features, feature_labels = [],[]
    uncut_seq, cut_site, left, right, ins_seq = indel_details
    for mh_min, mh_max in [(1,1),(2,2),(3,3),(4,6),(7,10),(11,15)]:
        feature_labels.append('L_MH%d-%d' % (mh_min, mh_max))
        features.append(any([hasLeftMH(left,right, uncut_seq, x) for x in range(mh_min, mh_max+1)]))
        feature_labels.append('R_MH%d-%d' % (mh_min, mh_max))
        features.append(any([hasRightMH(left,right, uncut_seq, x) for x in range(mh_min, mh_max+1)]))
        if mh_max > 2:
            feature_labels.append('L_MM1_MH%d-%d' % (mh_min, mh_max))
            features.append(any([hasLeftMH(left,right, uncut_seq, x, 1) for x in range(mh_min, mh_max+1)]))
            feature_labels.append('R_MM1_MH%d-%d' % (mh_min, mh_max))
            features.append(any([hasRightMH(left,right, uncut_seq, x, 1) for x in range(mh_min, mh_max+1)]))
    feature_labels.append('No MH'); features.append(sum(features)==0)
    if len(ins_seq) > 0:    #Apply these features to deletions only
        features = [0 for x in features]
    return features, feature_labels

def features_pairwise(features1, feature_labels1, features2, feature_labels2 ):
    pairwise_labels, pairwise_features = [], []
    for i,(val1,label1) in enumerate(zip(features1, feature_labels1)):
        for j,(val2,label2) in enumerate(zip(features2, feature_labels2)):
            pairwise_labels.append('PW_%s_vs_%s' % (label1, label2))
            pairwise_features.append(val1*val2)
    return pairwise_features, pairwise_labels

def calculateFeatures(indel_details):
    all_features, all_feature_labels = [], []
    feature_list = [feature_InsSize, feature_DelSize, feature_DelLoc, feature_InsLoc, feature_I1or2Rpt,feature_InsSeq, feature_LocalCutSiteSequence, feature_LocalCutSiteSeqMatches, feature_LocalRelativeSequence, features_SeqMatches, feature_microhomology]
    lin_fts = {x.__name__: x(indel_details) for x in feature_list }
    pairwise_list = [('feature_DelSize','feature_DelLoc'),('feature_InsSeq','feature_I1or2Rpt'), ('feature_microhomology','feature_DelSize')]
    pairwise_list += [('feature_microhomology','feature_DelLoc'), ('feature_LocalRelativeSequence','feature_DelSize'), ('feature_LocalCutSiteSequence','feature_InsSize')]
    pairwise_list += [('features_SeqMatches','feature_DelSize'),('feature_LocalRelativeSequence','feature_DelLoc'), ('feature_LocalCutSiteSequence','feature_DelSize')]
    pairwise_list += [('feature_LocalCutSiteSeqMatches','feature_DelSize'), ('feature_LocalCutSiteSequence','feature_I1or2Rpt')]
    pairwise_list += [('feature_LocalCutSiteSeqMatches','feature_I1or2Rpt')]
    for(fname1, fname2) in pairwise_list:
        features, feature_labels = features_pairwise(lin_fts[fname1][0],lin_fts[fname1][1],lin_fts[fname2][0],lin_fts[fname2][1])
        all_features.extend(features); all_feature_labels.extend(feature_labels)
    for fname in lin_fts:
        all_features.extend(lin_fts[fname][0]); all_feature_labels.extend(lin_fts[fname][1])
    assert(len(all_features)==len(all_feature_labels))
    return all_features, all_feature_labels

def calculate_features_for_gen_indel_file( generated_indel_file, uncut_seq, cut_site, out_file, is_reverse=False):
    len_seq = len(uncut_seq)
    f = io.open(generated_indel_file )
    f.readline()
    # fout = io.open(out_file, 'w')
    pam_dir = 'REVERSE' if is_reverse else 'FORWARD'
    # fout.write(u'###%s\t%d\t%s\n' % (uncut_seq, cut_site, pam_dir))
    first = True
    A,T,G,C = 'A','T','G','C'
    AA,AT,AC,AG,CG,CT,CA,CC = 'AA','AT','AC','AG','CG','CT','CA','CC'
    GT,GA,GG,GC,TA,TG,TC,TT = 'GT','GA','GG','GC','TA','TG','TC','TT'

    rows = []
    for toks in csv.reader(f,delimiter='\t'):
        indel, indel_locs = toks[0], eval(toks[2])
        for indel_loc in indel_locs:
            ins_seq = indel_loc[2] if len(indel_loc) > 2 else ''
            left = indel_loc[0] if not is_reverse else (len_seq - 1 - indel_loc[1]) 
            right = indel_loc[1] if not is_reverse else (len_seq - 1 - indel_loc[0]) 
            ins_seq = ins_seq if not is_reverse else Bio.Seq.reverse_complement(ins_seq)
            indel_details = (uncut_seq, cut_site, left, right, ins_seq)
             
            features, feature_labels = calculateFeatures(indel_details)
                
            # if first: fout.write(u'Indel\tLeft\tRight\tInserted Seq\t%s\n' % '\t'.join(feature_labels))
            # feature_str = '\t'.join(['%d' % x for x in features])
            # fout.write(u'%s\t%d\t%d\t%s\t%s\n' % (indel, left, right, ins_seq, feature_str))
            # first = False

            rows.append([indel, left, right, ins_seq] + features)
    columns = ['Indel', 'Left', 'Right', 'Inserted Seq'] + feature_labels   
    df = pd.DataFrame(rows, columns = columns)

    indel_feature_data = 1*df[['Indel'] + feature_labels].groupby('Indel').any()
    indel_feature_data['Indel'] = indel_feature_data.index

    indel_feature_data.loc[:,feature_labels] = indel_feature_data.loc[:,feature_labels].astype("float32")
    indel_feature_data.to_pickle(out_file)

    f.close()

def read_features_data(features_file):
    # feature_data = pd.read_csv(features_file, skiprows=2, sep='\t', dtype={'Inserted Seq':str})
    indel_feature_data = pd.read_pickle(features_file)
    feature_cols = [x for x in indel_feature_data.columns if x not in ['Oligo ID','Indel','Left','Right','Inserted Seq']]
    return indel_feature_data, feature_cols

def correct_inDelphi(o):
    o["TargetSequence"] = o["TargetSequence"] + "AGATCGGAAG"
    if ["Strand"] == "REVERSE":
        o["TargetSequence"] = "GTCAT" + o["TargetSequence"]
        o["PAM Index"] = o["PAM Index"] + 5
    return o

if __name__ == "__main__":

    datasets = [
        # ("FORECasT", "train"),
        # ("FORECasT", "test"),
        # ("FORECasT", "HAP1_test"),
        ("FORECasT", "TREX_A_test"),
        # ("LUMC", "WT"),
        # ("LUMC", "POLQ"),
        # ("LUMC", "KU80"),
        # ("LUMC", "LIG4"),
        # ("inDelphi", "0226-PRLmESC-Lib1-Cas9"),
        # ("inDelphi", "0105-mESC-Lib1-Cas9-Tol2-BioRep2-techrep1"),
        # ("inDelphi", "052218-U2OS-+-LibA-postCas9-rep1")
    ]

    output_dir = os.environ['OUTPUT_DIR']
    generated_indels_dir = output_dir + "model_training/data/FORECasT/indels/{}/"
    features_output_dir = output_dir + "model_training/data/FORECasT/features/{}/"

    if len(sys.argv) > 1:
        d = sys.argv[1]
        t = sys.argv[2]
        oligos = sys.argv[3:]
        
        if d == "FORECasT":
            oligo_f = t.split("_")[-1] + ".fasta"
        elif d == "inDelphi":
            oligo_f = "LibA.forward.fasta"
        else:
            oligo_f = t + ".forward.fasta"
        
        oligo_details = get_details_from_fasta("../../data/{}/{}".format(d, oligo_f))
        
        if d == "inDelphi":
            for o in oligo_details:
                oligo_details[o] = correct_inDelphi(oligo_details[o])
 
        for o in tqdm(oligos):
            gen_indel_f = generated_indels_dir.format(t) + o + "_genindels.txt"
            uncut_seq = oligo_details[o]["TargetSequence"]
            cut_site = oligo_details[o]["PAM Index"] - 3
            is_reverse = oligo_details[o]["Strand"] == "REVERSE"
            output_f = features_output_dir.format(t) + o
            calculate_features_for_gen_indel_file( gen_indel_f, uncut_seq, cut_site, output_f, is_reverse=is_reverse)
    else:
        for d, t in datasets:
            if d == "FORECasT":
                oligo_f = t.split("_")[-1] + ".fasta"
            elif d == "inDelphi":
                oligo_f = "LibA.forward.fasta"
            else:
                oligo_f = t + ".forward.fasta"

            oligo_details = get_details_from_fasta("../../data/{}/{}".format(d, oligo_f))
            os.makedirs(features_output_dir.format(t), exist_ok=True)
            batch_size = 250
            oligos = list(oligo_details.keys())
            oligos = [o.replace("(", "\(") for o in oligos]
            oligos = [o.replace(")", "\)") for o in oligos]
            for i in range(0, len(oligos), batch_size):
                print("sbatch features.batch {} {} {}".format(d, t, " ".join(oligos[i:i+batch_size])))
                os.system("sbatch features.batch {} {} {}".format(d, t, " ".join(oligos[i:i+batch_size])))



        
