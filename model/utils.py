import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
import math
from torch.utils.data import Dataset
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import pandas as pd
from tqdm import tqdm
from typing import Sequence, List, Union 
import pandas as pd 
from scipy.spatial.distance import cdist

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int = 15,
                 cycle_mult : float = 2,
                 max_lr : float = 0.1,
                 min_lr : float = 1e-6,
                 warmup_steps : int = 5,
                 gamma : float = 0.9,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class SVC_Dataset(Dataset):
    def __init__(self, data_ori, cell_morphology_vec, nuclear_morphology_vec, location, identity_vec=None):
        self.data_ori = data_ori
        self.cell_morphology_vec = cell_morphology_vec
        self.nuclear_morphology_vec = nuclear_morphology_vec
        self.location = location
        if identity_vec is not None:
            self.identity_vec = identity_vec
        
    def __len__(self):
        return len(self.data_ori)

    def __getitem__(self, index):
        if hasattr(self, 'identity_vec'):
            return  self.data_ori[index], self.cell_morphology_vec[index], self.nuclear_morphology_vec[index],  self.location[index], self.identity_vec[index]
        else:
            return  self.data_ori[index], self.cell_morphology_vec[index], self.nuclear_morphology_vec[index],  self.location[index]


def negative_binomial_loss(y_true, mu, r):
    y = y_true.float()
    
    term1 = torch.special.gammaln(y + r)
    term2 = torch.special.gammaln(y + 1)
    term3 = torch.special.gammaln(r)
    term4 = r * (torch.log(r) - torch.log(r + mu))
    term5 = y * (torch.log(mu) - torch.log(r + mu))
    
    log_likelihood = term1 - term2 - term3 + term4 + term5
    return -log_likelihood

def find_closest_point_preprocess(df1, df2, angle_col, type_col):
    ratio = []
    for _, row in df1.iterrows():
        angle_diffs = np.abs(df2[angle_col] - row[angle_col])
        idx = angle_diffs.argmin()
        df2_idx = df2.iloc[idx]
        
        distance_to_center2 = np.sqrt((row['x'] - df2_idx['x'])**2 + (row['y'] - df2_idx['y'])**2)
        ratio_i = row['distance_to_center']/(row['distance_to_center']+distance_to_center2)
        ratio.append(ratio_i)

    return ratio

def find_closest_point_postprocess(df1, df2, angle_col, type_col):
    closest_points = []
    for _, row in df1.iterrows():
        angle_diffs = np.abs(df2[angle_col] - row[angle_col])
        idx = angle_diffs.argmin()
        df2_idx = df2.iloc[idx]
        if df2_idx.empty:
            print("empty")
            continue
        closest_points.append(df2_idx['distance_to_center']*row['ratio'])

    return closest_points


def create_white_to_color_cmap(target_color, name='custom_cmap'):
    colors = [
        (1, 1, 1),         
        to_rgba(target_color)  
    ]
    return LinearSegmentedColormap.from_list(name, colors)


def create_color_cmap(source_color,target_color0, target_color=None, name='custom_cmap'):
    if target_color is None:
        colors = [
        to_rgba(source_color),         
        to_rgba(target_color0), 
    ]
    else:
        colors = [
            to_rgba(source_color),        
            to_rgba(target_color0),  
            to_rgba(target_color)  
        ]
    return LinearSegmentedColormap.from_list(name, colors)

def hsa2mmu(genelist: Sequence[str], drop: bool = False) -> Union[pd.DataFrame, List[str]]:
    from gseapy import Biomart

    genes = list(dict.fromkeys(genelist))  

    bm = Biomart()
    q = bm.query(
        dataset="hsapiens_gene_ensembl",
        attributes=[
            "external_gene_name",
            "ensembl_gene_id",
            "mmusculus_homolog_associated_gene_name",
            "mmusculus_homolog_ensembl_gene",
            "mmusculus_homolog_orthology_type",
            "mmusculus_homolog_perc_id",
        ],
        filters={"external_gene_name": genes},
    )

    if q is None or len(q) == 0:
        out = pd.DataFrame({"external_gene_name": genelist})
        out["ensembl_gene_id"] = None
        out["mmusculus_homolog_ensembl_gene"] = None
        out["mmusculus_homolog_associated_gene_name"] = None
        return out if not drop else []

    q["mmusculus_homolog_perc_id"] = pd.to_numeric(q["mmusculus_homolog_perc_id"], errors="coerce")

    def pick_best(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        one2one = df["mmusculus_homolog_orthology_type"].astype(str).str.contains("one2one", case=False, na=False)
        df["_one2one"] = one2one.astype(int)
        df["_pid"] = df["mmusculus_homolog_perc_id"].fillna(-1)
        return df.sort_values(["_one2one", "_pid"], ascending=[False, False]).head(1)

    best = q.groupby("external_gene_name", as_index=False, group_keys=False).apply(pick_best)
    best = best.drop(columns=["_one2one", "_pid"], errors="ignore")

    out = pd.DataFrame({"external_gene_name": genelist})
    out = out.merge(
        best[["external_gene_name", "ensembl_gene_id", "mmusculus_homolog_ensembl_gene", "mmusculus_homolog_associated_gene_name"]],
        on="external_gene_name",
        how="left",
    )

    if drop:
        return [x for x in out["mmusculus_homolog_associated_gene_name"].tolist() if isinstance(x, str) and x != ""]
    return out


def register_original_data(data_df, cell_contour_df):
    data_df['distance_to_center'] = np.sqrt((data_df['x'] - data_df['centerX'])**2 + (data_df['y'] - data_df['centerY'])**2)
    data_df['direction_vec'] = (np.degrees(np.arctan2(data_df['y'] - data_df['centerY'], data_df['x'] - data_df['centerX']))*2).round()/2
    data_df[ 'ratio'] = 0

    for cell in tqdm(data_df.cell.unique()):
        data_df['ratio'][data_df.cell==cell] = find_closest_point_preprocess(data_df[data_df.cell==cell], cell_contour_df[cell_contour_df.cell==cell], 'direction_vec', 'cell')

    data_df['angle_radians'] = data_df['direction_vec'].apply(lambda x: np.radians(x))
    data_df['x_norm'] = data_df.apply(lambda row: row['ratio'] * np.cos(row['angle_radians']), axis=1)
    data_df['y_norm'] = data_df.apply(lambda row: row['ratio'] * np.sin(row['angle_radians']), axis=1)
    print(data_df.head())
    
    return data_df

def process_gene(i_cell, cell, gene, df, map_height, map_width):#
    df_cell = df.loc[df['cell'] == cell] 
    x_max, x_min =  1,-1
    y_max, y_min =  1,-1

    x_bins_range = np.linspace(x_min-1e-10, x_max+1e-10, map_width +1)  
    y_bins_range = np.linspace(y_min-1e-10, y_max+1e-10, map_height +1)

    x_bin = pd.cut(df_cell['x_norm'], bins=x_bins_range, labels=False) 
    df_cell.loc[:,'x_norm_bin'] = x_bin
    y_bin = pd.cut(df_cell['y_norm'], bins=y_bins_range, labels=False)
    df_cell.loc[:,'y_norm_bin'] = y_bin
    df_fe = df_cell.loc[df_cell['gene'] == gene]
    map_fe = np.zeros((map_height, map_width))
    if not df_fe.empty:
        for idx in df_fe.index:
            idx_x = np.round(df_fe.loc[idx]['x_norm_bin']).astype(int)
            idx_y = np.round(df_fe.loc[idx]['y_norm_bin']).astype(int)
            if idx_x >= map_width or idx_y >= map_height or idx_x < 0 or idx_y < 0:
                print("idx_x, idx_y:",idx_x, idx_y,cell,fe)
                print("df_fe:",df_fe)

            map_fe[idx_y, idx_x] += 1
    return map_fe
     

def get_gene_map(df, cell_names, gene_names, map_height=48, map_width=48):
    # Load transcripts
    print("======> Loading transcripts file")
    print('%d unique genes' % len(gene_names))
    print('%d unique cells' % len(cell_names))
    print("Converting to expression maps")
    map_all_genes = np.zeros((len(cell_names), map_height, map_width, len(gene_names)), dtype=np.uint8)
    for i_cell, cell in enumerate(tqdm(cell_names)):
        for i_fe, fe in enumerate(gene_names):
                image = process_gene(i_cell, cell, fe, df, map_height, map_width)
                map_all_genes[i_cell, :, :, i_fe] = image.astype(np.uint8)
    # Save the combined map
    return map_all_genes
    print("Saved cell-gene map")

def compute_relative_dist_to_nuclear_center(image):

    total_sum = np.sum(image)
    if total_sum != 0:             
        dist = 0             
        for i in range(image.shape[0]):                 
            for j in range(image.shape[1]):                     
                dist += np.sqrt((i+0.5-6)**2 + (j+0.5-6)**2)/6 * image[i, j]              
        dist /= total_sum             
        dist = np.min([dist,1])          
    else:         
        dist = np.NaN
    
    return dist

def cal_gene_colocal_score(latent_np, gene_names, train_count_sum, gene_interested=None):

    if gene_interested is None:
        gene_interested = gene_names

    gene_idx = [gene_names.index(i)  for i in gene_interested]
    latent_np_interested = latent_np[:, gene_idx]
    
    most_similar_genes_all_cells = {gene: [] for gene in gene_interested}
    gene_colocal_score = np.zeros((len(gene_interested), len(gene_interested)))

    for cell in tqdm(range(len(latent_np))):
        # Compute pairwise distances for the current cell
        distances = cdist(latent_np_interested[cell], latent_np[cell], metric='cosine')
        # Identify the most similar genes for each gene of interest in the current cell
        idx = 0 
        for gene in gene_interested:
            if train_count_sum[cell][gene_idx][idx]!=0:
                most_similar_genes = np.argsort(distances[idx])[1:11]  # Exclude self-distance
                for j in most_similar_genes:
                    if gene_names[j] in gene_interested:
                        gene_colocal_score[gene_interested.index(gene), gene_interested.index(gene_names[j])] += 1
            idx += 1
    gene_colocal_score = gene_colocal_score/ ((train_count_sum!=0).sum(axis=0)[gene_idx]).reshape(-1,1).repeat(len(gene_interested), axis=1)
    gene_colocal_score_sym =  np.maximum(gene_colocal_score, gene_colocal_score.T)

    return gene_colocal_score_sym


def postprocess_sampling(prediction_mu, prediction_r, seed):
    np.random.seed(seed)
    count_data = np.zeros((prediction_mu.shape[0],prediction_mu.shape[1], 48, 48))
    for i in tqdm(range(prediction_mu.shape[0])):
        for j in range(prediction_mu.shape[1]):

            cell_gene_mu = prediction_mu[i,j]
            cell_gene_r = prediction_r[i,j]
            cell_gene_mu = np.repeat(cell_gene_mu/4, 4, axis=0)
            cell_gene_mu = np.repeat(cell_gene_mu/4, 4, axis=1)
            
            cell_gene_r = np.repeat(cell_gene_r, 4, axis=1)
            cell_gene_r = np.repeat(cell_gene_r, 4, axis=0)
            mask = cell_gene_r > 0
            count_data_0 = np.zeros((48,48))
            n = cell_gene_r[mask]
            p = cell_gene_r[mask] / (cell_gene_r[mask] + cell_gene_mu[mask])
            count_data_0[mask] = np.random.negative_binomial(n, p)
            count_data[i,j] = count_data_0
    return count_data


def postprocess_predictions(count_data, selected_gene, test_cell_names):
    predictions_pixel = pd.DataFrame()
    for i in selected_gene:
        count_data_i = count_data[:,selected_gene.index(i)]
        non_zero_indices = np.argwhere(count_data_i != 0)
        non_zero_values = count_data_i[non_zero_indices[:, 0],  non_zero_indices[:, 1], non_zero_indices[:, 2]]

        center_x, center_y = 24,24
        x, y = non_zero_indices[:, 2], non_zero_indices[:, 1]
        ratio = np.sqrt((x+0.5 - center_x) ** 2 + (y+0.5 - center_y) ** 2)/24
        ratio = np.minimum(ratio, 1)
        angles = (np.degrees(np.arctan2(y+0.5 - center_y, x+0.5 - center_x))*2).round()/2
        cell = np.array(test_cell_names)[non_zero_indices[:, 0]]
        gene = np.repeat(i, len(non_zero_values))
        count = non_zero_values.flatten()
        predictions_pixel_i = pd.DataFrame({
            'cell': cell,
            'gene': gene,
            'count': count,
            'x': x,
            'y': y,
            'ratio': ratio,
            'direction_vec': angles
        })

        predictions_pixel = pd.concat([predictions_pixel, predictions_pixel_i], axis=0)

    return predictions_pixel


def postprocess_predictions_original(predictions_pixel, df_cell_contour):
    predictions_pixel[['centerX', 'centerY', 'distance_to_center']] = 0

    for cell in tqdm(predictions_pixel.cell.unique()):
        cell_center_x = df_cell_contour.centerX[df_cell_contour.cell == cell].values[0]
        cell_center_y = df_cell_contour.centerY[df_cell_contour.cell == cell].values[0]

        mask = predictions_pixel["cell"] == cell

        dist = find_closest_point_postprocess(
            predictions_pixel.loc[mask],
            df_cell_contour.loc[df_cell_contour["cell"] == cell],
            "direction_vec",
            "cell",
        )

        predictions_pixel.loc[mask, ["centerX", "centerY"]] = [cell_center_x, cell_center_y]
        predictions_pixel.loc[mask, "distance_to_center"] = dist

    predictions_pixel['angle_radians'] = predictions_pixel['direction_vec'].apply(lambda x: np.radians(x))
    predictions_pixel['x_original'] = predictions_pixel.apply(lambda row: row['distance_to_center'] * np.cos(row['angle_radians']) + row['centerX'], axis=1)
    predictions_pixel['y_original'] = predictions_pixel.apply(lambda row: row['distance_to_center'] * np.sin(row['angle_radians']) + row['centerY'], axis=1)

    return predictions_pixel