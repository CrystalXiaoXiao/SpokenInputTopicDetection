import numpy as np
import sys

def convert_bounds_2_labels(boundary_seg):
    label = 0

    label_seg = []

    idx = 0
    while idx < len(boundary_seg):
        label_seg.append(label)
        if boundary_seg[idx] == 1:
            label += 1

        idx+=1

    return label_seg

def calculate_pk(ref,hyp):
    ref = convert_bounds_2_labels(ref)
    hyp = convert_bounds_2_labels(hyp)

    num_segs = ref[-1] + 1
    num_elems = len(ref)
    #print "Total number of segments: %i"%num_segs
    #print "Total number of units: %i"%num_elems
    p_k = 0
    k = int((0.5*num_elems/num_segs) - 1 )
    #print "k value: %i"%k

    if k == 0:
        k = 2

    for i in range(0,num_elems - k + 1):
        delta_ref = (ref[i] is ref[i+k - 1])
        delta_hyp = (hyp[i] is hyp[i+k - 1])

        if delta_ref != delta_hyp:
            p_k += 1
    p_k = p_k / float(num_elems - k + 1)

    return p_k

def evaluate_model(path_ref, doc_lengths, path_hyp, decision_confidence=0.5):
    ref_file = open(path_ref, 'r')
    hyp_file = open(path_hyp, 'r')
    doc_length = open(doc_lengths, 'r')
    doc_length = doc_length.readlines()
    doc_length = [int(l) for l in doc_length]

    p_k_values = []

    ref = ref_file.readlines()
    ref = [int(r.strip()) for r in ref]

    hyp = hyp_file.readlines()
    hyp = [1 if float(h.strip()) >= decision_confidence else 0 for h in hyp]

    total = 0
    for l in doc_length:
        start = total
        end = start + l
        total += l

        doc_ref = ref[start:end]
        doc_hyp = hyp[start:end]

        p_k_val = calculate_pk(doc_ref, doc_hyp)
        p_k_values.append(p_k_val)
        #print p_k_val

    print "%.4f"%np.mean(p_k_values),
    #print "%.4f"%calculate_pk(ref, hyp),

def main():
    ref_file = sys.argv[1]
    hyp_files = [#"./lda/out_bnds_lda_top1.txt",
                 #"./lda/out_bnds_lda_top3.txt",
                 #"./lda/out_bnds_lda_top7.txt",
                 #"./lda/out_bnds_lda_smooth.txt",
                 #"./embeddings/out_bnds_embeddings.txt",
                 #"./test_hyp_embed50.txt",
                 #"./combined.txt",
                 #"./combined3.txt",
                 #"./lda_embed_model.txt",
                sys.argv[2]
                #"./test_post_lda_small.txt",
                #"./test_combined_fulltrain_0.1alphaembed_0.9betalda.txt",
                #"./test_combined_fulltrain_0.3alphaembed_0.7betalda.txt",
                #"./test_combined_fulltrain_0.5alphaembed_0.5betalda.txt",
                #"./test_combined_fulltrain_0.7alphaembed_0.3betalda.txt",
                #"./test_combined_fulltrain_0.9alphaembed_0.1betalda.txt",
                #"./test_combined_0.1alphaembed_0.9betalda.txt",
                #"./test_combined_0.3alphaembed_0.7betalda.txt",
                #"./test_combined_0.5alphaembed_0.5betalda.txt",
                #"./test_combined_0.7alphaembed_0.3betalda.txt",
                #"./test_combined_0.9alphaembed_0.1betalda.txt"
                #"./combined_0.1alphaembed_0.9betalda.txt",
                #"./combined_0.3alphaembed_0.7betalda.txt",
                #"./combined_0.5alphaembed_0.5betalda.txt",
                #"./combined_0.7alphaembed_0.3betalda.txt",
                #"./combined_0.9alphaembed_0.1betalda.txt",
                #"./out_hyp_lda25.txt",
                #"./out_hyp_lda50.txt",
                #"./out_hyp_lda100.txt",
                #"./out_hyp_lda150.txt",
                #"./out_hyp_lda200.txt"
                 ]

    feat_path = sys.argv[3]
    if sys.argv[7] == "dev":
        doc_lengths = "%s/doc_length_window%d_max%d_%s_%s.txt"%(feat_path, int(sys.argv[4]), int(sys.argv[5]), sys.argv[7], sys.argv[6])
    else:
        doc_lengths = "%s/doc_length_window%d_max%d_%s_%s.txt"%(feat_path, int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7])

    for confidence in np.arange(0.1,0.9,0.1):
        print "Confidence: %.1f"%confidence
        for model in hyp_files:
            evaluate_model(ref_file, doc_lengths, model, confidence)
            print "\t",
        print "\n",

if __name__ == "__main__":
    main()

