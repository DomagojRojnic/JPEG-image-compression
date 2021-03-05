import numpy as np
import re

codes = {}

def zigzag(matrix):
    rows = matrix.shape[0]
    columns = matrix.shape[1]
    solution=[[] for i in range(rows+columns-1)] 
  
    for i in range(rows): 
        for j in range(columns): 
            sum=i+j 
            if(sum%2 ==0): 
                solution[sum].insert(0,matrix[i][j])
            else: 
                solution[sum].append(matrix[i][j]) 
    dum = []
    for i in solution:
        for j in i:
            dum.append(j)
    return dum

def reverse_zigzag(array, N):
    num_cols, num_rows = N-1, N-1
    out = np.zeros((N, N))
    tot_elem = len(array)
    cur_row = 0
    cur_col = 0
    cur_index = 0

    while cur_index < tot_elem:
        if cur_row==0 and (cur_row+cur_col)%2==0 and cur_col!=num_cols:
            out[cur_row,cur_col]=array[cur_index]
            cur_col=cur_col+1
            cur_index=cur_index+1
		
        elif cur_row==num_rows and (cur_row+cur_col)%2!=0 and cur_col!=num_cols:
            out[cur_row,cur_col]=array[cur_index]
            cur_col=cur_col+1
            cur_index=cur_index+1
            
        elif cur_col==0 and (cur_row+cur_col)%2!=0 and cur_row!=num_rows:
            out[cur_row,cur_col]=array[cur_index]
            cur_row=cur_row+1
            cur_index=cur_index+1
            
        elif cur_col==num_cols and (cur_row+cur_col)%2==0 and cur_row!=num_rows:
            out[cur_row,cur_col]=array[cur_index]
            cur_row=cur_row+1
            cur_index=cur_index+1
            
        elif cur_col!=0 and cur_row!=num_rows and (cur_row+cur_col)%2!=0:
            out[cur_row,cur_col]=array[cur_index]
            cur_row=cur_row+1		
            cur_col=cur_col-1
            cur_index=cur_index+1
            
        elif cur_row!=0 and cur_col!=num_cols and (cur_row+cur_col)%2==0:
            out[cur_row,cur_col]=array[cur_index]
            cur_row=cur_row-1		
            cur_col=cur_col+1
            cur_index=cur_index+1
            
        elif cur_index==tot_elem-1:
            out[num_rows,num_cols]=array[cur_index]
            break
    return out

def frequency(array):
    frequency = {}
    for element in array:
        if element not in frequency:
            frequency[element] = 1
        else:
            frequency[element] += 1
    return frequency

def sortFrequencies(frequencies):
    sortedFrequencies = dict(sorted(frequencies.items(), key=lambda item: item[1]))
    return [(k, v) for v, k in sortedFrequencies.items()] 

def buildTree(tuples):
    while len(tuples) > 1:
        leastTwo = tuple(tuples[0:2])
        theRest  = tuples[2:]
        combFreq = leastTwo[0][0] + leastTwo[1][0]
        tuples   = theRest + [(combFreq,leastTwo)]
        tuples.sort(key=lambda t: t[0])
    return tuples[0]

def trimTree(tree):
    p = tree[1]
    if type(p) == np.int32: 
        return p
    else: 
        return (trimTree(p[0]), trimTree(p[1]))

def assignCodes(node, pat=''):
    global codes
    if type(node) == np.int32:
        codes[node] = pat
    else:
        assignCodes(node[0], pat+"0")
        assignCodes(node[1], pat+"1")

def encode(array):
    global codes
    output = ""
    for element in array:
            output += str(codes[element])
    return output

def decode (tree, str) :
    output = []
    p = tree
    for bit in str:
        if bit == '0':
            p = p[0]
        else: 
            p = p[1]
        if type(p) == np.int32:
            output.append(p)
            p = tree
    return output

def encode_array(array):
    freqs = frequency(array)
    tuples = sortFrequencies(freqs)
    tree = buildTree(tuples)
    trim = trimTree(tree)
    assignCodes(trim)
    return encode(array), trim      