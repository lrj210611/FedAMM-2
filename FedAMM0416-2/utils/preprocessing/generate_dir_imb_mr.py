import csv
import os
import numpy as np
import torch
import random
import math
import pandas as pd
import matplotlib.pyplot as plt

seed = 1024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
iid = False
alpha = 0.01
modal_alpha = alpha
dir_name = 'dir_brats_split/'+str(seed)+'_'+str(alpha)
def main():
    np.random.seed(seed)
    masks_test = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
    
    n_masks = len(masks_test) 
    client_masks, client_mask_counts = generate_mask_distribution_only_maskid(
        alpha=alpha,
        modal_alpha=modal_alpha,
        masks_test=masks_test
    )
    if iid:
        client_masks_one = np.array([0]*3+[1]*3+[2]*3+[3]*3+[4]*3+[5]*4+[6]*4+[7]*4+[8]*4+[9]*4+[10]*4+[11]*4+[12]*4+[13]*4+[14]*4)
        client_masks = np.array([client_masks_one,client_masks_one,client_masks_one,client_masks_one])
        client_mask_counts_one = np.array([3,3,3,3,3,4,4,4,4,4,4,4,4,4,4])
        client_mask_counts = np.array([client_mask_counts_one,client_mask_counts_one,client_mask_counts_one,client_mask_counts_one])

    currentdirPath = os.path.dirname(os.path.abspath(__file__))
    relativePath = '../../datalist'
    datarootPath = os.path.abspath(os.path.join(currentdirPath,relativePath))
    train_path = os.path.join(datarootPath, 'BRATS2020_Training_none_npy')
    split_path = os.path.join(datarootPath, dir_name)
    if os.path.exists(split_path):
        raise FileExistsError(f"目录 '{split_path}' 已存在！请先删除该目录或使用其他路径。")
    os.makedirs(split_path, exist_ok=True)
    train_file1 = os.path.join(train_path, 'client_train_split_seed_1234/client_part_1.txt')
    csv_name1 = os.path.join(split_path, 'client_part_1_imb.csv')
    train_file2 = os.path.join(train_path, 'client_train_split_seed_1234/client_part_2.txt')
    csv_name2 = os.path.join(split_path, 'client_part_2_imb.csv')
    train_file3 = os.path.join(train_path, 'client_train_split_seed_1234/client_part_3.txt')
    csv_name3 = os.path.join(split_path, 'client_part_3_imb.csv')
    train_file4 = os.path.join(train_path, 'client_train_split_seed_1234/client_part_4.txt')
    csv_name4 = os.path.join(split_path, 'client_part_4_imb.csv')

    client_modal_weight = []
    mask_id_count_list = []
    client_mask_id_proportions = []
    clinet_modal_num1,mask_id_count1 = gengeate_imb_file(train_file1, csv_name1, client_masks[0])
    clinet_modal_num2,mask_id_count2 = gengeate_imb_file(train_file2, csv_name2, client_masks[1])
    clinet_modal_num3,mask_id_count3 = gengeate_imb_file(train_file3, csv_name3, client_masks[2])
    clinet_modal_num4,mask_id_count4 = gengeate_imb_file(train_file4, csv_name4, client_masks[3])
    client_modal_weight.append(clinet_modal_num1)
    client_modal_weight.append(clinet_modal_num2)
    client_modal_weight.append(clinet_modal_num3)
    client_modal_weight.append(clinet_modal_num4)
    mask_id_count_list.append(mask_id_count1)
    mask_id_count_list.append(mask_id_count2)
    mask_id_count_list.append(mask_id_count3)
    mask_id_count_list.append(mask_id_count4)
    total = [sum(x) for x in zip(*mask_id_count_list)]
    for mask_id_count in mask_id_count_list:
        client_mask_id_proportions.append([mask_id_count[i] / total[i] if total[i] != 0 else 0 for i in range(len(mask_id_count))])
    client_mask_proportions_sum = [sum(client_mask_id_proportions[i])/len(client_mask_id_proportions[i]) for i in range(len(client_mask_id_proportions))]
    
    
    client_modal_weight = np.array(client_modal_weight)
    all_client_modal_weight = client_modal_weight.sum(axis=0)
    all_client_modal_weight = np.where(all_client_modal_weight == 0, 1, all_client_modal_weight)
    client_modal_weight = client_modal_weight / all_client_modal_weight

    log_file = os.path.join(split_path, 'distribution_log.txt')
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f'client_mask_proportions_sum:\n {client_mask_proportions_sum}\n')
        f.write(f'client_mask_id_proportions:\n {client_mask_id_proportions}\n')
        f.write(f'client_modal_weight:\n {client_modal_weight}\n')
        f.write(f'client_1:clinet_modal_num : {clinet_modal_num1}\n mask_id_count : {mask_id_count1}\n')
        f.write(f'client_2:clinet_modal_num : {clinet_modal_num2}\n mask_id_count : {mask_id_count2}\n')
        f.write(f'client_3:clinet_modal_num : {clinet_modal_num3}\n mask_id_count : {mask_id_count3}\n')
        f.write(f'client_4:clinet_modal_num : {clinet_modal_num4}\n mask_id_count : {mask_id_count4}\n')
        infor_vsi(masks_test, alpha, n_masks, client_masks, client_mask_counts,split_path,f)

def infor_vsi(masks_test, alpha, n_masks, client_masks, client_mask_counts,split_path,f):
    for client_id in range(4):
        unique, counts = np.unique(client_masks[client_id], return_counts=True)
        f.write(f"\nClient {client_id}:")
        f.write(f"Unique mask_ids: {sorted(unique)}\n")
        f.write(f"Counts: {counts}\n")
        f.write(f"Total samples: {len(client_masks[client_id])}\n")
        
        missing_masks = set(range(n_masks)) - set(unique)
        if missing_masks:
            f.write(f"Warning: Missing mask_ids: {missing_masks}\n")
        
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs))
        f.write(f"Distribution entropy: {entropy:.3f}\n")
        flair,t1,t1ce,t2 = 0,0,0,0
        for m in client_masks[client_id]:
            if masks_test[m][0]: flair += 1
            if masks_test[m][1]: t1 += 1
            if masks_test[m][2]: t1ce += 1
            if masks_test[m][3]: t2 += 1
        f.write(f"Modal counts: FLAIR:{flair}, T1:{t1}, T1CE:{t1ce}, T2:{t2}\n")
        f.write(f"Miss modal ratio: {round(1-flair/len(client_masks[client_id]),3)},{round(1-t1/len(client_masks[client_id]),3)},{round(1-t1ce/len(client_masks[client_id]),3)},{round(1-t2/len(client_masks[client_id]),3)}\n")
    
    visualize_distribution(client_mask_counts, alpha,split_path)
    visualize_modal_distribution(client_masks, masks_test,split_path)

def gengeate_imb_file(train_file, csv_name, p):
    with open(train_file, 'r') as f:
        datalist = [i.strip() for i in f.readlines()]

    img_max = len(datalist)

    index = 0
    pos_index = []

    mask_array = np.array([[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
            [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
            [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
            [True, True, True, True]])
    file = open(csv_name, "a+")
    csv_writer = csv.writer(file)
    csv_writer.writerow(['data_name', 'mask_id', 'mask', 'pos_mask_ids'])
    for i in range(img_max):
        index = p[i]
        mask = mask_array[index]
        if np.array_equal(mask, [False, False, True, False]):
            pos_index = [2]
        elif np.array_equal(mask,[False, True, False, False]):
            pos_index = [1]
        elif np.array_equal(mask, [True, False, False, False]):
            pos_index = [3]
        elif np.array_equal(mask, [False, False, False, True]):
            pos_index = [0]
        elif np.array_equal(mask, [False, True, True, False]):
            pos_index = [1,2,5]
        elif np.array_equal(mask, [True, False, True, False]):
            pos_index = [2,3,6]
        elif np.array_equal(mask, [False, False, True, True]):
            pos_index = [0,2,7]
        elif np.array_equal(mask, [True, True, False, False]):
            pos_index = [1,3,9]
        elif np.array_equal(mask, [False, True, False, True]):
            pos_index = [0,1,4]
        elif np.array_equal(mask, [True, False, False, True]):
            pos_index = [0,3,8]
        elif np.array_equal(mask, [True, True, True, False]):
            pos_index = [1,2,3,5,6,9,10]
        elif np.array_equal(mask, [False, True, True, True]):
            pos_index = [0,1,2,4,5,7,13]
        elif np.array_equal(mask, [True, False, True, True]):
            pos_index = [0,2,3,6,7,8,11]
        elif np.array_equal(mask, [True, True, False, True]):
            pos_index = [0,1,3,4,8,9,12]
        elif np.array_equal(mask, [True, True, True, True]):
            pos_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    
        csv_writer = csv.writer(file)
        csv_writer.writerow([datalist[i],index,[mask[0],mask[1],mask[2],mask[3]],pos_index])
    file.close()
    
    imb_mr_csv_data = pd.read_csv(csv_name)
    clinet_modal_num = np.zeros(4, dtype=np.float32)
    for sample_mask in imb_mr_csv_data['mask']:
        clinet_modal_num += np.array(eval(sample_mask), dtype=np.float32)
    print("clinet_modal_num",clinet_modal_num)
    mask_id_count = [0] * 15
    for mask_id in imb_mr_csv_data['mask_id']:
        mask_id_count[mask_id] += 1
    print('Mask ID Count: {}'.format(mask_id_count))
    return clinet_modal_num,mask_id_count

def generate_mask_distribution_only_modal(n_clients=4, samples_per_client=55, n_masks=15, alpha=0.1, modal_alpha=0.05, masks_test=None):
    client_masks = [[] for _ in range(n_clients)]
    
    remaining_samples = samples_per_client
    
    modal_proportions = np.random.dirichlet(np.repeat(modal_alpha, 4), size=n_clients)
    
    for client_id in range(n_clients):
        modal_weights = np.zeros(n_masks)
        for mask_id in range(n_masks):
            mask = masks_test[mask_id]
            modal_weight = 0
            for modal_idx, has_modal in enumerate(mask):
                if has_modal:
                    modal_weight += modal_proportions[client_id][modal_idx]
            modal_weights[mask_id] = modal_weight
        
        combined_probs = modal_weights
        probs = combined_probs / combined_probs.sum()
        
        additional_masks = np.random.choice(n_masks, 
                                          size=remaining_samples, 
                                          p=probs)
        client_masks[client_id].extend(additional_masks)
    
    client_masks = [np.array(masks) for masks in client_masks]
    client_mask_counts = np.zeros((n_clients, n_masks), dtype=int)
    
    for client_id in range(n_clients):
        for mask_id in range(n_masks):
            client_mask_counts[client_id, mask_id] = np.sum(client_masks[client_id] == mask_id)
    
    return client_masks, client_mask_counts

def generate_mask_distribution_only_maskid(n_clients=4, samples_per_client=55, n_masks=15, alpha=0.1, modal_alpha=0.05, masks_test=None):
    client_masks = [[] for _ in range(n_clients)]
    
    for client_id in range(n_clients):
        for mask_id in range(n_masks):
            client_masks[client_id].append(mask_id)
    
    remaining_samples = samples_per_client - n_masks
    
    mask_proportions = np.random.dirichlet(np.repeat(alpha, n_masks), size=n_clients)

    for client_id in range(n_clients):
        base_probs = mask_proportions[client_id]

        combined_probs = base_probs
        probs = combined_probs / combined_probs.sum()
        
        additional_masks = np.random.choice(n_masks, 
                                          size=remaining_samples, 
                                          p=probs)
        client_masks[client_id].extend(additional_masks)
        np.random.shuffle(client_masks[client_id])
    
    client_masks = [np.array(masks) for masks in client_masks]
    client_mask_counts = np.zeros((n_clients, n_masks), dtype=int)
    
    for client_id in range(n_clients):
        for mask_id in range(n_masks):
            client_mask_counts[client_id, mask_id] = np.sum(client_masks[client_id] == mask_id)
    
    return client_masks, client_mask_counts



def generate_mask_distribution(n_clients=4, samples_per_client=55, n_masks=15, alpha=0.1, modal_alpha=0.05, masks_test=None):
    client_masks = [[] for _ in range(n_clients)]
    
    for client_id in range(n_clients):
        for mask_id in range(n_masks):
            client_masks[client_id].append(mask_id)
    
    remaining_samples = samples_per_client - n_masks
    
    modal_proportions = np.random.dirichlet(np.repeat(modal_alpha, 4), size=n_clients)
    
    mask_proportions = np.random.dirichlet(np.repeat(alpha, n_masks), size=n_clients)
    
    for client_id in range(n_clients):
        base_probs = mask_proportions[client_id]
        
        modal_weights = np.zeros(n_masks)
        for mask_id in range(n_masks):
            mask = masks_test[mask_id]
            modal_weight = 0
            for modal_idx, has_modal in enumerate(mask):
                if has_modal:
                    modal_weight += modal_proportions[client_id][modal_idx]
            modal_weights[mask_id] = modal_weight
        
        combined_probs = base_probs * (modal_weights + 0.1)
        probs = combined_probs / combined_probs.sum()
        
        additional_masks = np.random.choice(n_masks, 
                                          size=remaining_samples, 
                                          p=probs)
        client_masks[client_id].extend(additional_masks)
        np.random.shuffle(client_masks[client_id])
    
    client_masks = [np.array(masks) for masks in client_masks]
    client_mask_counts = np.zeros((n_clients, n_masks), dtype=int)
    
    for client_id in range(n_clients):
        for mask_id in range(n_masks):
            client_mask_counts[client_id, mask_id] = np.sum(client_masks[client_id] == mask_id)
    
    return client_masks, client_mask_counts


def visualize_modal_distribution(client_masks, masks_test,split_path):
    n_clients = len(client_masks)
    modal_counts = np.zeros((n_clients, 4))
    
    for client_id, masks in enumerate(client_masks):
        for mask_id in masks:
            for modal_idx, has_modal in enumerate(masks_test[mask_id]):
                if has_modal:
                    modal_counts[client_id, modal_idx] += 1
    
    plt.figure(figsize=(10, 6))
    modalities = ['FLAIR', 'T1', 'T1CE', 'T2']
    x = np.arange(n_clients)
    width = 0.2
    colors = ['#DE582B', '#1868B2', '#018A67', '#F3A332']
    for i in range(4):
        plt.bar(x + i*width, modal_counts[:, i], width, label=modalities[i],color=colors[i])
    
    plt.ylabel('Number of modalities')
    plt.title(f'Distribution of Modalities across Clients (alpha= {alpha})')
    plt.legend()
    plt.xticks(x + width*1.5, [f'Client {i+1}' for i in range(n_clients)])
    plt.tight_layout()
    plt.savefig(os.path.join(split_path, 'modal_distribution.png'))
    plt.show()

def visualize_distribution(client_mask_counts, alpha,split_path):
    n_clients, n_masks = client_mask_counts.shape
    
    plt.figure(figsize=(10, 6))
    bottom = np.zeros(n_clients)
    colors = plt.cm.tab20.colors
    for mask_id in range(n_masks):
        plt.bar(range(n_clients), client_mask_counts[:, mask_id],
                bottom=bottom, label=f'Mask {mask_id}',color=colors[mask_id])
        bottom += client_mask_counts[:, mask_id]
    
    plt.ylabel('Number of samples')
    plt.title(f'Distribution of Modalities Mask across Clients (alpha={alpha})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(range(n_clients), [f'Client {i+1}' for i in range(n_clients)])
    plt.tight_layout()
    plt.savefig(os.path.join(split_path, 'mask_distribution.png'))
    plt.show()



if __name__ == '__main__':
    main()
    print('generate_dir_imb_mr.py done!')
