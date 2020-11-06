from fastdtw import fastdtw
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import partial

def MSE(array1, array2):
    if array1.shape != array2.shape:
        raise ValueError('The shapes of input arrays are \
            not equal:{} / {}'.format(array1.shape, array2.shape))

    if array1.ndim == 2:
        diff = array1 - array2
        mse = np.mean(np.sum(diff ** 2, axis=1))
    elif array1.ndim == 1:
        diff = array1 - array2
        mse = np.sum(diff ** 2)
    else:
        raise ValueError('Dimension must be [T, dim] or [dim].')

    return mse

def dur_dtw(m_a,m_b,t_a=None,t_b=None):
    if t_a is not None:
        m_a=m_a[:t_a,:]
        m_b=m_b[:t_b,:]
    result=fastdtw(m_a,m_b,1,MSE)
    return result

def get_align_matrix(target_1,target_2,index_1,index_2):
    result_1=target_1.index_select(0,torch.LongTensor(index_1).to(target_1.device))
    result_2=target_2.index_select(0,torch.LongTensor(index_2).to(target_2.device))
    result=torch.stack([result_1,result_2],dim=-1)

    return result

#  just ues in inference
# def caculat_dtw_loss(a,b,index_1,index_2):
#     target_1=a
#     target_2=b
#     with ProcessPoolExecutor(max_workers=8) as executor:
#         result=executor.map(get_align_matrix,target_1,target_2,index_1,index_2)
#     result=list(result)
#
#     return result

def dtw_loss(a,b,index_1,index_2,Loss=torch.nn.MSELoss()):
    results=[]
    for x,y,z,n in zip(a,b,index_1,index_2):
        results.append(get_align_matrix(x,y,z,n))
    loss=0
    for result in  results:
        loss+=Loss(result[:,:,0],result[:,:,1])
    return loss/len(results)






def multi_process_path(a,b,dur_a,dur_b):
    input_a=a.detach().cpu().numpy()
    input_b=b.detach().cpu().numpy()
    d_a=dur_a.detach().cpu().numpy()
    d_b=dur_b.detach().cpu().numpy()

    with ProcessPoolExecutor(max_workers=8) as executor:
        result=executor.map(dur_dtw,input_a,input_b,d_a,d_b)
    index_a=[]
    index_b=[]
    for batch in result:
        index_a_ = []
        index_b_ = []
        for index in batch[1]:
            index_a_.append(index[0])
            index_b_.append(index[1])
        index_a.append(index_a_)
        index_b.append(index_b_)
    return index_a,index_b

if __name__=='__main__':

        import time

        a = torch.from_numpy(np.random.randn(3, 500, 80)).cuda()
        b = torch.from_numpy(np.random.randn(3, 500, 80)).cuda()
        a.requires_grad = True
        b.requires_grad = True
        s = time.time()
        index_a, index_b = multi_process_path(a, b,[100,99,90],[99,80,70])
        print('*********')
        result=dwt_loss(a,b,index_a,index_b)
        e=time.time()
        print(e-s)









