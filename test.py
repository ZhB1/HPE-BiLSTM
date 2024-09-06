import utils
import os
import torch
import numpy as np
np.set_printoptions(suppress=True)

role_name = ['introductionSen',  
             'thesisSen',
             'ideaSen',
             'exampleSen',   
             'conclusionSen',  
             'otherSen',
             'evidenceSen']
csv_head = 'name, accuracy, all-p, all-r, all-f, macro-f, micro-f'
for n in role_name:
    for p in ['-p', '-r', '-f']:
        csv_head += ', ' + n + p
        
role_name_e = ['MajorClaim',  
             'Claim',
             'Premise',
             'Other']
csv_head_e = 'name, accuracy, all-p, all-r, all-f, macro-f, micro-f'
for n in role_name_e:
    for p in ['-p', '-r', '-f']:
        csv_head_e += ', ' + n + p


def PRF(a, ignore=[]):
    precision = []
    recall = []
    f1 = []
    real = []
    TP = 0
    TPFP = 0
    TPFN = 0

    for i in range(len(a[0])):
        precision.append(a[i][i] / sum(a[:, i]))
        recall.append(a[i][i] / sum(a[i]))
        f1.append((precision[i] * recall[i] * 2) / (precision[i] + recall[i]))
        real.append(sum(a[i]))
        if i not in ignore:
            TP += a[i][i]
            TPFP += sum(a[:, i])
            TPFN += sum(a[i])

    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1 = np.nan_to_num(f1)
    real = np.nan_to_num(real)
    print(precision)
    print(recall)
    print(f1)
    

    a_p = 0
    a_r = 0
    a_f = 0
    m_p = TP / TPFP
    m_r = TP / TPFN
    
    for i in range(len(f1)):
        if i not in ignore:
            a_p += real[i] * precision[i]
            a_r += real[i] * recall[i]
            a_f += real[i] * f1[i]

    total = sum(real) - sum(real[ignore])
    # print('test', total, a_p)
    print(a_p/total, a_r/total, a_f/total)
    
    macro_f = np.delete(f1,ignore,0).mean()    
    micro_f = (m_p * m_r * 2) / (m_p + m_r)
    print(macro_f, micro_f)
    # print(m_p, m_r)
    
    all_prf = [m_r, a_p/total, a_r/total, a_f/total, macro_f, micro_f]
    return precision, recall, f1, all_prf

def MeanError(a):
    n = len(a)
    MSE = 0.
    MAE = 0.
    for i in range(n):
        for j in range(n):
            if not i == j:
                MSE += (i - j)**2 * a[i][j]
                MAE += abs(i - j) * a[i][j]
    c = sum(sum(a))
    MSE = MSE / c
    MAE = MAE / c
    # print(sum(a))
    print(MSE, MAE)
    return MSE, MAE

def test_all(test, newdir, w_file, data, title=False, is_mask=False):
    if len(data) == 3:
        pad_documents, pad_labels, features = data
    else:
        pad_documents, pad_labels, features, ctx_vecs = data
    with open(w_file,'w',encoding='utf-8') as wf:
        wf.write(csv_head + '\n')
        filenames = os.listdir(newdir)
        for file in filenames:
            fname = os.path.join(newdir, file)
            print(file)
            tag_model = torch.load(fname, map_location='cpu')  
    #         tag_model.pWeight = torch.nn.Parameter(torch.ones(3))

            if len(data) == 3:
                accuracy, a = test(tag_model, pad_documents, pad_labels, features, 'cpu', batch_n=1, title=title, is_mask=is_mask)
            else:
                accuracy, a = test(tag_model, pad_documents, pad_labels, features, ctx_vecs, 'cpu', batch_n=1, title=title, is_mask=is_mask)
            print(accuracy)
            print(a)
            
            precision, recall, f1, all_prf = PRF(a[:-1, :-1], ignore=[5])
            accuracy, all_p, all_r, weighted_f, macro_f, micro_f = all_prf
            
            wf.write('_'.join(file.split('_')[: -1]))
            wf.write(', ' + str(accuracy))
            wf.write(', ' + str(all_p) + ', ' + str(all_r) + ', ' + str(weighted_f))            
            wf.write(', ' + str(macro_f))
            wf.write(', ' + str(micro_f))
            for i in range(len(f1)):
                wf.write(', ' + str(precision[i]) + ', ' +  str(recall[i]) + ', ' + str(f1[i]))

            wf.write('\n')

def Chinese_test(model_dir):
    in_file = './data/Ch_test.json'
    embed_filename = './embd/new_embeddings2.txt'
    title = True

    print('load Embeddings...')
    max_len = 40
    embed_map, vec_size = utils.loadEmbeddings(embed_filename)

    en_documents, en_labels, features, vec_size = utils.getSamplesAndFeatures(in_file, embed_filename, title=title)
    pad_documents, pad_labels = utils.sentence_padding(en_documents, en_labels, max_len, vec_size)
    
    is_mask = False
    
    from train import test
    w_file = './value/%s_%s.csv' % (in_file.split('.')[1].split('/')[-1], model_dir.split('/')[2])
    test_all(test, model_dir, w_file, (pad_documents, pad_labels, features), title, is_mask=is_mask)
    