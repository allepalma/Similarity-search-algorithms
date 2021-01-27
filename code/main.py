import os
import numpy as np
import scipy as scipy
from scipy import sparse
from scipy.sparse import find
import time
from sklearn.metrics.pairwise import paired_cosine_distances
import itertools
import sys
import argparse

#Set up the reading of the input from the terminal
parser = argparse.ArgumentParser(description = 'Find pairs of similar users')
parser.add_argument('-d', '--directory', type=str, required=True)
parser.add_argument('-s', '--seed', type=int, required=True)
parser.add_argument('-m', '--method', type=str, required=True, choices = ['js','cs','dcs'])
args = parser.parse_args()



'''Jaccard similarity: minhashing'''

def Jaccard_minhashing(input_mat, sig_size):
    #Set the number of rows and number of columns
    n_col, n_row = input_mat.shape[1], input_mat.shape[0]
    #Create a sparse signature matrix with all zeroes and turn to lil format
    sig = sparse.csr_matrix((sig_size, n_col))
    sig = sig.tolil()
    #Set up the vector containing the indexes of the rows of the sparse matrix
    pos = np.arange(n_row)
    #Initialize the indices and inptr of the input matrix
    ind = input_mat.indices
    indptr = input_mat.indptr
    for iteration in range(sig_size):
        #Shuffle the indexes of the rows at random and re-arrange the matrix as a perm_mat
        np.random.shuffle(pos)
        #Initialize vector containing the indexes of the already filled entries of the signature matrix
        #at the current permutation (empty at the beginning)
        col_non_null_sig = np.array([])
        #i is the index of the row of the permutation considered.
        i = 0
        #Until we still have some non-null columns in the signature
        while len(col_non_null_sig) < n_col:
            #Fetch the columns that are non-null at the row of the permutation considered
            col_non_null_new = ind[indptr[pos[i]]:indptr[pos[i]+1]]
            #The columns to update are the ones that are non-null in the evaluated row and that are still null in
            #the signature
            col_update = np.setdiff1d(col_non_null_new, col_non_null_sig)
            #Then update the columns that in the signature are not null anymore
            col_non_null_sig = np.union1d(col_non_null_sig, col_update)
            #Update the signature with the value of the row examined
            sig[iteration, col_update] = i+1
            i+=1
    return sig

def Jaccard_similarity(mat1,mat2):
    #Implement the jaccard similarity for two matrices. The columns of the matrices represent vectors of users to compare
    #such that we must compute the Jaccard similarity of the column i of mat1 and mat2.
    #The elementwise multiplication preserves as 1's the entries that were 1 in both matrices. The rest is 0.
    #We sum across the rows the nummber of elements that were 1 in both matrices.
    equal = np.array(np.sum(mat1.multiply(mat2), axis=0))[0]
    tot = np.array(np.sum(mat1!=mat2, axis=0)[0])+equal
    return np.array(equal/tot)[0]



'''Cosine sketching'''

def cosine_sketching(input_mat, sig_size):
    #Initialize the number of rows and number of columns
    n_col, n_row = input_mat.shape[1], input_mat.shape[0]
    #Create a sparse signature matrix with all zeroes
    sig = sparse.csr_matrix((sig_size, n_col))
    sig = sig.tolil()    
    for iteration in range(sig_size):
        random_vec = np.random.uniform(-1,1, n_row).reshape((-1,1))
        sig_row = np.sign(input_mat.T.dot(random_vec))
        sig[iteration,:] = sig_row.T
    return sig

def Cosine_similarity(mat1, mat2):
    #Transpose the matrices since the pairwise model applied on two matrices
    #operates on the rows as pairs of vectors.
    cos_dist = paired_cosine_distances(mat1.T, mat2.T)
    cos_sim = 1 - cos_dist
    theta = np.arccos(cos_sim)
    theta = theta * (180/np.pi)
    return 1-theta/180



'''Comparison of signatures'''

def sign_comparison(mat1, mat2, sig_size):
    #This function has the scope to compute the degree of similarity between two non-binary matrices
    #ehose columns represent signatures. 
    diff = mat1 != mat2 
    sums = np.sum(diff,axis=0)
    return 1-(sums/sig_size)


'''The LSH algorithm'''

def LSH(signature, b, r):
    #We initialize the buckets as an empty dictionary
    buckets = {}
    for i in range(b):
        #Setup the row range
        row_range = [r*i, r*i+r]
        for j in range(signature.shape[1]):
            #Fix the blocks for all columns
            block = np.array(signature[row_range[0]:row_range[1],j].todense()).ravel()
            val = ''.join(map(str,block))
            hashed = hash(val)
            #The keys are tuples of the block number and the hashed value associated to it
            buckets[(i, hashed)] = buckets.get((i, hashed),[])+[j]
    return buckets


'''The check for similar pairs'''

def pairs(buck):
    #Given a bucket of feasible size, isolate all possible pairs in two lists,
    #one for the first elements of all pairs and one for the elements of the other pairs
    pairs1, pairs2 = [],[]
    comb = itertools.combinations(buck,2)
    for combination in comb:
        pairs1.append(combination[0])
        pairs2.append(combination[1])
    return pairs1, pairs2


def similar_pairs(bucket, signature, input_mat, upper, sig_size, sim):
    #Remove buckets with a single observation and also those with multiple observations exceeding
    #the threshold
    bucks = [i for i in bucket.values() if len(i)>1 and len(i)<=upper]
    #The two lists will contain the pairs whose signatures are more similar than the
    #threshold.
    pair1, pair2 = [],[]
    for i in bucks:
        p1, p2 = pairs(i)
        #Create the two matrices of signatures with the pairs of the buckets
        mat_sig1 = signature[:,p1]
        mat_sig2 = signature[:,p2]
        score_sign = np.array(sign_comparison(mat_sig1, mat_sig2, sig_size))[0]
        for i in range(len(score_sign)):
            if score_sign[i]>=threshold:
                pair1.append(p1[i])
                pair2.append(p2[i])  
    mat_pair1 = input_mat[:,pair1]
    mat_pair2 = input_mat[:,pair2]
    similarities = sim(mat_pair1,mat_pair2)
    count=0
    #Make sure there are no duplicate pairs.
    visited = []
    for i in range(len(similarities)):
        if similarities[i]>=threshold:
            pair = sorted((pair1[i],pair2[i]))
            if pair not in visited:
                file_object = open(m+'.txt', 'a+')
                #You sum by 1 since we had downsized the pairs by 1 before, but in the representation of the real dataset the range starts
                #from 1 (non-python indexes).
                file_object.write(str(pair[0]+1) + ',' + str(pair[1]+1)+'\n')
                file_object.close()
                count += 1
                visited.append(pair)
    print('The number of found pairs is: ', count)
    return


if __name__=='__main__':
    d = args.directory
    s = int(args.seed)
    m = args.method


    #Overwrite existing file with same name
    file = open(m+'.txt', 'w')
    file.close()
                
    
    ratings = np.load(d)

    if m == 'js' or m == 'dcs':
        ratings[:,2] = 1


    #Renumber users and movies with indexes from 0 to the maximum user and movie -1 to create the sparse matrix correctly
    ratings[:,0] = ratings[:,0]-1
    ratings[:,1] = ratings[:,1]-1

    
    #Convert to a sparse matrix the input pandas data frame
    input_mat = sparse.csr_matrix((ratings[:,2], (ratings[:,1], ratings[:,0])), shape=(17770, 103703))

    
    if m == 'js':       
        sig_size = 150 
        b = 25
        r = 6
        upper_bond = 100
        
        threshold = 0.5
        np.random.seed(s)
        print('Start Minhasing')
        start_tot = time.time()
        start = time.time()
        minh = Jaccard_minhashing(input_mat, sig_size)
        end = time.time()
        print('The runtime for minhashing with signature size ',sig_size,' is ',end-start, ' seconds')
        print('Start LSH')
        np.random.seed(s)
        start = time.time()
        buckets = LSH(minh, b, r)
        end = time.time()
        print('The runtime for LSH with b = ', b ,' and r = ', r , end-start, ' seconds')
        print('Evaluate the pairs')
        sim = Jaccard_similarity
        start = time.time()
        similarity = similar_pairs(buckets, minh ,input_mat, upper_bond, sig_size, sim)
        end = time.time()
        print('The runtime for pair assessment is ',end-start)
        end_total = time.time()
        print('Total elapsed: ', end_total-start_tot, ' seconds')

    else:
        threshold = 0.73
        if m == 'cs':
            sig_size = 250 
            b = 12
            r = 20
            upper_bond = 100
        elif m == 'dcs':
            sig_size = 250 
            b = 13
            r = 18
            upper_bond = 100
            
        np.random.seed(s)
        print('Start sketching')
        start_tot = time.time()
        start = time.time()
        minh = cosine_sketching(input_mat, sig_size)
        end = time.time()
        print('The runtime for sketching with signature size ',sig_size,' is ',end-start, ' seconds')
        print('Start LSH')
        start = time.time()
        buckets = LSH(minh, b, r)
        end = time.time()
        print('The runtime for LSH with b = ', b ,' and r = ', r , end-start, ' seconds')
        print('Evaluate the pairs')
        sim = Cosine_similarity
        start = time.time()
        similarity = similar_pairs(buckets, minh ,input_mat, upper_bond, sig_size, sim)
        end = time.time()
        print('The runtime for pair assessment is ',end-start, ' seconds')
        end_total = time.time()
        print('Total elapsed: ', end_total-start_tot, ' seconds')




