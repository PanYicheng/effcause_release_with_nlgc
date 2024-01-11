"""
Root Cause Analysis Functions.
"""
from collections import defaultdict
import numpy as np
from scipy import stats


def normalize_by_column(transition_matrix):
    for col_index in range(transition_matrix.shape[1]):
        if np.sum(transition_matrix[:, col_index]) == 0:
            continue
        transition_matrix[:, col_index] = transition_matrix[:, col_index] / np.sum(
            transition_matrix[:, col_index]
        )
    return transition_matrix


def bfs(
    transition_matrix, entry_point, reach_end=True, max_path_length=None, verbose=False
):
    """ Backtrace breadth first search in a graph.

    Params:
        transition_matrix: the transition matrix of the graph.
        entry_point: the entry point of the bfs, the frontend service.
        reach_end: If False, BFS stops when encoutering the same node.
                   If True, BFS doesn't stop until there is no previous node.
        max_path_length: the maximum path length. None means no limit.
        verbose: whether print detailed information.
    """
    path_list = set()
    queue = [[entry_point - 1]]
    while len(queue) > 0:
        # Limit output path list size to 10000 in case of infinite bfs
        if len(path_list) > 10000:
            break
        # Limit bfs queue size to 10000 in case of infinite bfs, and flush paths to path_list
        if len(queue) > 10000:
            while len(queue) > 0:
                path_list.add(tuple(queue.pop(0)))
            break
        if verbose and verbose>=2:
            # verbose level >= 2: print BFS queue info
            print(
                "{space:^15}BFS Queue Len:{i:>6d} Output queue size: {:>6d}"
                .format(len(path_list), space="", i=len(queue)),
                end="\r",
            )
            
        path = queue.pop(0)
        if np.sum(transition_matrix[:, path[-1]]) == 0:
            # if there is no previous node, the path ends and we add it to output.
            path_list.add(tuple(path))
        else:
            # if there is at least one previous node
            if max_path_length is not None and len(path) >= max_path_length:
                # if path length exceeds limit
                path_list.add(tuple(path))
            else:
                # Try extending the path with every possible node
                for prev_node in range(transition_matrix.shape[0]):
                    if transition_matrix[prev_node, path[-1]] > 0.0 and (
                        prev_node not in path
                    ):
                        # extend the path
                        new_path = path + [prev_node]
                        queue.append(new_path)
                    elif transition_matrix[prev_node, path[-1]] > 0.0 and not reach_end:
                        # if encounter repeated node, stop bfs if reach_end is set to False.
                        path_list.add(tuple(path))
    if verbose and verbose>=2:
        # verbose level >= 2: print BFS queue info endline
        print("")
    return path_list


def search_path(
    transition_matrix,
    mean_method="harmonic",
    max_path_length=None,
    entry_point=14,
    verbose=False,
):
    """
    Seach for anomaly propagation paths based on transition matrix. 
    The pathes will be sorted by the existence probability.
    
    """
    path_list = bfs(
        transition_matrix,
        entry_point,
        reach_end=True,
        max_path_length=max_path_length,
        verbose=verbose,
    )
    # use different mean methods to estimate the path existence probability.
    path_list_prob = []
    for path in path_list:
        p = []
        end = path[0]
        for start in path[1:]:
            p.append(transition_matrix[start, end])
            end = start
        if len(p) == 0:
            path_list_prob.append(0)
        else:
            if mean_method == "arithmetic":
                path_list_prob.append(np.mean(p))
            elif mean_method == "geometric":
                # Remove probability equal to 1 because they
                # don't contain useful information.
                p = [_ for _ in p if _ != 1]
                path_list_prob.append(stats.gmean(p))
            elif mean_method == "harmonic":
                # Remove probability equal to 1 because they
                # don't contain useful information.
                p = [_ for _ in p if _ != 1]
                path_list_prob.append(stats.hmean(p))

    # sort path by descending probability
    out = [item for item in zip(path_list_prob, path_list)]
    out.sort(key=lambda x: x[0], reverse=True)
    # verbose level >= 2: print 10 backward paths in BFS at most
    if verbose and verbose>=2:
        for i in out[:10]:
            print("{:^5}{:<5.4f},{}".format("", i[0], [_ + 1 for _ in i[1]]))
    return out

def ranknode(
    data, out_path, entry_point, node_num, topk_path=60, prob_thres=0.4, num_sel_node=1
):
    """Rank node according to the anomaly score which includes both
    path level correlation and pearson correlation.
    """
    # Select possible root cause nodes from path
    path_node_count = defaultdict(int)
    # select only first topk paths with prob >= threshold
    for i in out_path[:topk_path]:
        for node in i[1][-num_sel_node:]:
            path_node_count[node] = path_node_count[node] + 1
        if i[0] < prob_thres:
            break
    # exclude entry point
    if entry_point - 1 in path_node_count:
        path_node_count.pop(entry_point - 1)

    # Calculate correlation between selected node and entry point
    path_node_corr = {}
    for node in path_node_count:
        ret = np.corrcoef(
            np.concatenate(
                [data[:, entry_point - 1].reshape(1, -1), data[:, node].reshape(1, -1)],
                axis=0,
            )
        )
        #     print('Node:{} Corr:{}'.format(node, abs(ret[0, 1])))
        path_node_corr[node] = abs(ret[0, 1])

    # Estimate node root cause score according to both
    # path count and correlation
    rank_list = []
    for node in path_node_count:
        rank_list.append(
            [
                node,
                path_node_count[node] * 1.0 / (num_sel_node * topk_path)
                + path_node_corr[node] * 1.0,
            ]
        )
    rank_list.sort(key=lambda x: x[1], reverse=True)

    # TODO: Find a method to include the other nodes. For example, score them 
    # with correlation coefficient and the path covering count.
    # append all other nodes according to correlation coefficient
    # other_node = []
    # for node in range(node_num):
    #     if node not in candidate_node:
    #         ret = np.corrcoef(
    #             np.concatenate(
    #                 [data[:, entry_point-1].reshape(1, -1),
    #                  data[:, node].reshape(1, -1)], axis=0))
    #         other_node.append([node, abs(ret[0, 1])])
    # other_node.sort(key=lambda x: x[1], reverse=True)
    # candidate.extend(other_node)
    return rank_list


def analyze_root(
    transition_matrix,
    entry_point,
    local_data,
    mean_method="harmonic",
    max_path_length=None,
    topk_path=150,
    prob_thres=0.4,
    num_sel_node=3,
    use_new_matrix=False,
    verbose=False,
):
    """Perform the root cause analysis based on the generated dependency graph.

    Args:
        transition_matrix : 
        entry_point : 
        local_data : 
        epoch : . Defaults to 1000.
        mean_method (str, optional): [description]. Defaults to "harmonic".
        max_path_length ([type], optional): [description]. Defaults to None.
        topk_path (int, optional): [description]. Defaults to 150.
        prob_thres (float, optional): [description]. Defaults to 0.4.
        num_sel_node (int, optional): [description]. Defaults to 3.
        use_new_matrix (bool, optional): [description]. Defaults to False.
        verbose (bool, optional): [description]. Defaults to False.

    Returns:
        ranked_nodes : A list of tuples (node, score).
    """
    out_path = search_path(
        transition_matrix,
        mean_method=mean_method,
        max_path_length=max_path_length,
        entry_point=entry_point,
        verbose=verbose,
    )
    ranked_nodes = ranknode(
        local_data,
        out_path,
        entry_point,
        local_data.shape[1],
        topk_path=topk_path,
        prob_thres=prob_thres,
        num_sel_node=num_sel_node,
    )
    # adjust node representation. Node'id ranges from 1 to N.
    for j in range(len(ranked_nodes)):
        ranked_nodes[j][0] += 1
    return ranked_nodes