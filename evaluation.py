import numpy as np
import pandas as pd
import cdt
from sklearn.metrics import accuracy_score, f1_score

from cdt.metrics import SHD
from utils.graph_utils import df_to_cpdag_pcalg

def evaluate_ad_pairs(ad_pairs: list, dag_df: pd.DataFrame):
    def has_path(start, end, dag_df):
        # BFS to check if a path exists from start to end
        visited = set()
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node == end:
                return True
            visited.add(node)
            # Get all children of node (i.e., outgoing edges)
            children = dag_df.columns[(dag_df.loc[node] == 1)].tolist()
            for child in children:
                if child not in visited:
                    queue.append(child)
        return False

    correct = 0
    for ancestor, descendant in ad_pairs:
        if has_path(ancestor, descendant, dag_df):
            correct += 1
    return correct / len(ad_pairs) if ad_pairs else 0

def parse_class(df, A, B):
    if df.at[A, B] == 1:
        return 1
    elif df.at[B, A] == 1:
        return 2
    else:
        return 0

def get_all_edges(df):
    # Get all positions where value is 1
    positions = [(idx, col) for idx, col in zip(*np.where(df.values == 1))]

    # Convert numerical indices to the corresponding index and column labels
    positions = [(df.index[row], df.columns[col]) for row, col in positions]

    return positions


def evaluate_pairwise(dag_df, pred_df):
    variables = list(dag_df.columns)
    n_variables = len(variables)

    ground_truth = set(get_all_edges(dag_df))
    prediction = set(get_all_edges(pred_df))
    
    correct = ground_truth & prediction
    
    precision = len(correct) / len(prediction)
    recall = len(correct) / len(ground_truth)
    
    f1_score = 2 * precision * recall / (precision + recall)
    
    accuracy = accuracy_score(y_true=dag_df, y_pred=pred_df)
    return accuracy, f1_score

            
def evaluate_generation(dag_df, pred_df, logger):
    logger.debug("Evaluating generation...")
    
    pred = pred_df.to_numpy().astype(int)
    dag = dag_df.to_numpy().astype(int)
    n_variables = len(dag_df.columns)
    
    logger.debug("Start counting corrects...")
    corrects = np.equal(pred, dag).astype(int)
    
    num_corr = np.sum(corrects) - n_variables
    
    tp = np.sum((pred == 1) & (dag == 1))
    pred_sum = np.sum(pred)
    dag_sum = np.sum(dag)

    logger.debug("Start calculating ...")
    precision = tp / pred_sum if pred_sum != 0 else 0
    recall = tp / dag_sum if dag_sum != 0 else 0
    logger.debug(f"Precision: {precision}, Recall: {recall}")
    
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall)/(precision + recall)
    logger.debug(f"F1 Score: {f1_score}")
    
    accuracy = num_corr/(pred.size - n_variables)
    logger.debug(f"Accuracy: {accuracy}")
    
    hamming_distance = np.sum(pred != dag)
    logger.debug(f"Hamming Distance: {hamming_distance}")
    nhd = hamming_distance/(n_variables * n_variables)
    logger.debug(f"NHD: {nhd}")
    
    cpdag_np = df_to_cpdag_pcalg(dag_df).to_numpy().astype(int)
    pred_cpdag_np = df_to_cpdag_pcalg(pred_df).to_numpy().astype(int)
    logger.debug("DAG and Pred DAG converted to CPDAG in numpy array")
    
    shd_cdt = SHD(cpdag_np, pred_cpdag_np, False)
    logger.debug(f"SHD in CPDAG: {shd_cdt}")
    
    return precision, recall, f1_score, accuracy, nhd, shd_cdt
