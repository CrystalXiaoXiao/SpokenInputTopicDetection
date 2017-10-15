import sys

def align_file(file_path, marker_path, sen_size, shift):
    dev_ref = []

    file_marker = open(marker_path,'r')
    file_marker = file_marker.readlines()
    file_marker = [int(line.strip()) for line in file_marker]

    with open(file_path) as ref_file:
        id = 0
        for line, mark in zip(ref_file,file_marker):
            line = line.strip()
            line = line.split()
            line = [float(w) for w in line]
            if mark == 1:
                id = len(dev_ref)
                dev_ref += line
            else:
                for i in range(sen_size - shift):
                    #if line[i] > dev_ref[id + i]:
                    dev_ref[id+i] += line[i]

                dev_ref += line[sen_size-shift:]
            id += shift
    return dev_ref

def align_seq(sen_size = 18, shift = 18, feat_path="", suffix=""):
    dev_ref = []
    dev_hyp = []
    overlap = sen_size/shift

    dev_ref = align_file("dev_target.txt", "%s/marker_sen%i_shift%i_%s.txt"%(feat_path,sen_size, shift, suffix), sen_size, shift)
    dev_hyp = align_file("dev_hyp.txt", "%s/marker_sen%i_shift%i_%s.txt"%(feat_path,sen_size, shift, suffix), sen_size, shift)

    dev_ref_n = [str(num/overlap) for num in dev_ref]
    dev_hyp_n = [str(num/overlap) for num in dev_hyp]

    print len(dev_ref_n)
    print len(dev_hyp_n)
    dev_f = open(sys.argv[1],'w')
    dev_h = open(sys.argv[2], 'w')

    dev_f.write('\n'.join(dev_ref_n))
    dev_h.write('\n'.join(dev_hyp_n))



align_seq(int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6])
