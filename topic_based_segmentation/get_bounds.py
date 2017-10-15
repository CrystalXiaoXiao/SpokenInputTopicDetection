import sys
with open(sys.argv[1]) as file:
    for line in file.readlines():
        line = line.replace('[','').replace(']','').strip()

        line = [float(num) for num in line.split()]

        #if line[1] >= float(sys.argv[1]):
        #    print 1
        #else:
        #    print 0
        print line[1]