import sys
import numpy as np

def get_posteriors(path):
    post = []
    with open(path) as file:
        for line in file.readlines():
            # line = line.replace('[','').replace(']','')
            # line = line.strip()
            # line = [float(num) for num in line.split()]
            # #if line[1] >= 0.5:
            # #    print 1
            # #else:
            # #    print 0
            post.append(float(line))
    return post

def combine_models(models_post,models_exp_weights):
    combined = []

    for sample in range(len(models_post[0])):
        p_c_all = 1

        for model, exp_weight in zip(models_post, models_exp_weights):
            p_c_all *= (model[sample] ** exp_weight)

        combined.append(p_c_all)

    return combined

def main():
    model1 = get_posteriors(sys.argv[1])
    model2 = get_posteriors(sys.argv[2])

    models = [model1, model2]
    for alpha in np.arange(0.1, 1.0, 0.2):
        weights = [alpha, 1 - alpha]
        combined = combine_models(models, weights)
        with open("./results/merged_%.1f_%.1f.txt"%(alpha,1-alpha),'wb') as f:
            f.write('\n'.join([str(c) for c in combined]))


if __name__ == "__main__":
    main()