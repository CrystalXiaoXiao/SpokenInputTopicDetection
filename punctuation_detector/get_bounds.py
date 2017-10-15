import sys
with open('tmp_dev_hyp.txt') as file:
    for line in file.readlines():
        line = line.replace('[','').replace(']','')
        line = [float(num) for num in line.split()]
        if line[0] >= float(sys.argv[1]):
            print 1
        else:
            print 0
        #print line[1]
