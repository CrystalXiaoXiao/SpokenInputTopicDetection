import sys
target = open('tmp_dev_target.txt')
hyp = open(sys.argv[1])

target = target.readlines()
hyp = hyp.readlines()

total_pred = 0
total_bnds = 0
total_non_bnds = 0
true_pos = 0
false_pos = 0
true_neg = 0
false_neg = 0
total = 0

for i, pred in enumerate(hyp):
    pred = float(pred.strip())
    real = float(target[i].strip())
    if real > 0:
	real = 1

    total += 1
    if real == 1:
        total_bnds += 1
    if pred == 1:
        total_pred += 1
    if real == 1:
        total_non_bnds += 1
    if (pred == real) and real == 1:
        true_pos += 1
        #elif (i-2>=0) and (hyp[i-2].strip() == 1):
        #    true_pos += 1
        #elif (i+2 < len(target)) and hyp[i+2].strip() == 1:
        #    true_pos += 1
        #else:
        #    false_pos += 1
    if (pred == real) and real == 0:
        true_neg += 1
    if (pred == 1) and (real != 1):
        skew = True
        if (i-1>=0) and float(hyp[i-1].strip()) == 1 and float(target[i-1].strip() != 1):
            true_pos += 1
        elif (i+1 < len(target)) and float(hyp[i+1].strip()) == 1 and float(target[i+1].strip()) != 1:
            true_pos += 1
        elif (i-2>=0) and (float(hyp[i-2].strip()) == 1) and (float(target[i-2].strip()) != 1):
            true_pos += 1
        elif (i+2 < len(target)) and float(hyp[i+2].strip()) == 1 and float(target[i+2].strip()) != 1:
            true_pos += 1
        # elif (i-3>=0) and (float(hyp[i-3].strip()) == 1):
        #     true_pos += 1
        # elif (i+3 < len(target)) and float(hyp[i+3].strip()) == 1:
        #     true_pos += 1
        # elif (i-4>=0) and (float(hyp[i-4].strip()) == 1):
        #     true_pos += 1
        # elif (i+4 < len(target)) and float(hyp[i+4].strip()) == 1:
        #     true_pos += 1
        else:
            skew = False
            false_pos += 1

    if (pred != 1) and (real == 1):
        skew = True
        if (i-1>=0) and float(hyp[i-1].strip()) == 1 and float(target[i-1].strip() != 1):
            true_pos += 1
        elif (i+1 < len(target)) and float(hyp[i+1].strip()) == 1 and float(target[i+1].strip()) != 1:
            true_pos += 1
        elif (i-2>=0) and (float(hyp[i-2].strip()) == 1) and (float(target[i-2].strip()) != 1):
            true_pos += 1
        elif (i+2 < len(target)) and float(hyp[i+2].strip()) == 1 and float(target[i+2].strip()) != 1:
            true_pos += 1
        # elif (i-3>=0) and (float(hyp[i-3].strip()) == 1):
        #     true_pos += 1
        # elif (i+3 < len(target)) and float(hyp[i+3].strip()) == 1:
        #     true_pos += 1
        # elif (i-4>=0) and (float(hyp[i-4].strip()) == 1):
        #     true_pos += 1
        # elif (i+4 < len(target)) and float(hyp[i+4].strip()) == 1:
        #     true_pos += 1
        else:
            skew = False
            false_neg += 1
        #if skew:
        #    false_neg -= 1

precision = true_pos*100/(1.0*true_pos+1.0*false_pos)
recall = true_pos*100/(1.0*true_pos+1.0*false_neg)
print "Total boundaries: %i"%(total_bnds)
print "Predicted boundaries: %i"%(total_pred)
print "Precision: %.2f"%(precision)
print "Recall: %.2f"%(recall)
print "F1-score: %.2f"%((2*precision*recall)/(1.0*(precision + recall)))
print "Acc: %.2f"%((true_pos+true_neg)*100/(1.0*total))
