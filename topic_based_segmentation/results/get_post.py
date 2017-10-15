with open('test_hyp_lda150.txt') as file:
    for line in file.readlines():
        line = line.replace('[','').replace(']','')
        line = [float(num) for num in line.split()]
        #if line[1] >= 0.5:
        #    print 1
        #else:
        #    print 0
        print line[1]