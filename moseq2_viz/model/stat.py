import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import f1_score, confusion_matrix

def get_session_mean_df(df, stat, max_syllable):
    '''

    Parameters
    ----------
    df
    stat
    max_syllable

    Returns
    -------

    '''

    df_pivot = (
        df[df.syllable < max_syllable]
            .pivot_table(index=["group", "uuid"], columns="syllable", values=stat)
            .replace(np.nan, 0)
    )

    return df_pivot

def bootstrap_me(usages, n_iters=10000):
    '''

    Parameters
    ----------
    usages
    n_iters

    Returns
    -------

    '''

    n_mice = usages.shape[0]
    return np.nanmean(usages[np.random.choice(n_mice, size=(n_mice, n_iters))], axis=0)

def ztest_vect(d1, d2):
    '''

    Parameters
    ----------
    d1
    d2

    Returns
    -------

    '''

    mu1 = d1.mean(0)
    mu2 = d2.mean(0)
    std1 = d1.std(0)
    std2 = d2.std(0)
    std = np.sqrt(std1 ** 2 + std2 ** 2)
    return np.minimum(1.0, 2 * stats.norm.cdf(-np.abs(mu1 - mu2) / std))

def bootstrap_group_means(df, group1, group2, statistic, max_syllable):
    '''

    Parameters
    ----------
    df
    group1
    group2
    statistic
    max_syllable

    Returns
    -------

    '''

    # get separated group variables
    group_stat = get_session_mean_df(df, statistic, max_syllable)

    groups = (group1, group2)
    usages = {k: group_stat.loc[k].values for k in groups}
    boots = {k: bootstrap_me(v) for k, v in usages.items()}

    return boots

def get_tie_correction(x, N_m):
    '''

    Parameters
    ----------
    x
    N_m

    Returns
    -------

    '''

    vc = x.value_counts()
    tie_sum = 0
    if (vc > 1).any():
        tie_sum += np.sum(vc[vc != 1] ** 3 - vc[vc != 1])
    return tie_sum / (12. * (N_m - 1))

def run_manual_KW_test(df_usage, merged_usages_all, num_groups, n_per_group, cum_group_idx, n_perm, shape, dof, seed=0):
    '''

    Parameters
    ----------
    df_usage
    merged_usages_all
    num_groups
    n_per_group
    cum_group_idx
    n_perm
    shape
    dof
    seed

    Returns
    -------

    '''

    # create random index array n_perm times
    rnd = np.random.RandomState(seed=seed)
    perm = rnd.rand(n_perm, shape[0]).argsort(-1)

    real_ranks = np.apply_along_axis(stats.rankdata, 0, merged_usages_all)
    X_ties = df_usage.apply(get_tie_correction, 0, N_m=shape[0]).values
    KW_tie_correct = np.apply_along_axis(stats.tiecorrect, 0, real_ranks)

    # rank data
    perm_ranks = real_ranks[perm]

    # get square of sums for each group
    ssbn = np.zeros((n_perm, shape[1]))
    for i in range(num_groups):
        ssbn += perm_ranks[:, cum_group_idx[i]:cum_group_idx[i + 1]].sum(1) ** 2 / n_per_group[i]

    # h-statistic
    h_all = 12.0 / (shape[0] * (shape[0] + 1)) * ssbn - 3 * (shape[0] + 1)
    h_all /= KW_tie_correct
    p_vals = stats.chi2.sf(h_all, df=dof)

    # check that results agree
    p_i = np.random.randint(n_perm)
    s_i = np.random.randint(shape[1])
    kr = stats.kruskal(*np.array_split(merged_usages_all[perm[p_i, :], s_i], np.cumsum(n_per_group[:-1])))
    assert (kr.statistic == h_all[p_i, s_i]) & (kr.pvalue == p_vals[p_i, s_i]), "manual KW is incorrect"

    return h_all, real_ranks, X_ties

def plot_H_stat_significance(df_k_real, h_all, N_s):
    '''

    Parameters
    ----------
    df_k_real
    h_all
    N_s

    Returns
    -------

    '''

    fig, ax = plt.subplots(figsize=(9, 6))

    pcts = np.percentile(h_all, [2.5, 97.5], axis=0)

    ax.fill_between(np.arange(N_s), pcts[0, :], pcts[1, :], alpha=0.2, color="k")

    ax.scatter(np.arange(N_s)[~df_k_real.is_sig], df_k_real.statistic[~df_k_real.is_sig], label='Not significant')
    ax.scatter(np.arange(N_s)[df_k_real.is_sig], df_k_real.statistic[df_k_real.is_sig], label='Significant Syllable')

    ax.set_xlabel("Syllable")
    ax.set_ylabel(f"$H$")
    ax.legend()
    sns.despine()

    return fig, ax

def run_kruskal(df, statistic, max_syllable, n_perm=10000, seed=42, THRESH=0.05, MC_METHOD='fdr_bh'):
    '''

    Parameters
    ----------
    df
    statistic
    max_syllable
    n_perm
    seed
    THRESH
    MC_METHOD

    Returns
    -------

    '''

    rnd = np.random.RandomState(seed=seed)

    # get mean grouped data
    grouped_data = get_session_mean_df(df, statistic, max_syllable).reset_index()

    # KW Constants
    vc = grouped_data.group.value_counts().loc[grouped_data.group.unique()]
    n_per_group = vc.values
    group_names = vc.index

    cum_group_idx = np.insert(np.cumsum(n_per_group), 0, 0)
    num_groups = len(group_names)
    dof = num_groups - 1

    df_only_usage = grouped_data[range(max_syllable)]
    merged_usages_all = df_only_usage.values

    N_m, N_s = merged_usages_all.shape

    h_all, real_ranks, X_ties = run_manual_KW_test(df_usage=df_only_usage,
                                                   merged_usages_all=merged_usages_all,
                                                   num_groups=num_groups,
                                                   n_per_group=n_per_group,
                                                   cum_group_idx=cum_group_idx,
                                                   n_perm=n_perm,
                                                   shape=merged_usages_all.shape,
                                                   dof=dof,
                                                   seed=seed)

    df_k_real = pd.DataFrame([stats.kruskal(*np.array_split(merged_usages_all[:, s_i],
                                                            np.cumsum(n_per_group[:-1]))) for s_i in range(N_s)])

    df_k_real['emp_fdr'] = multipletests(((h_all > df_k_real.statistic.values).sum(0) + 1) / n_perm,
                                         alpha=THRESH, method=MC_METHOD)[1]

    df_k_real['is_sig'] = df_k_real['emp_fdr'] < THRESH
    print(f"Found {df_k_real['is_sig'].sum()} syllables that pass threshold {THRESH} with {MC_METHOD}")

    null_zs_within_group, real_zs_within_group = permute_within_group_pairs(grouped_data,
                                                                            vc,
                                                                            real_ranks,
                                                                            X_ties,
                                                                            N_m,
                                                                            group_names,
                                                                            rnd,
                                                                            n_perm)

    df_pair_corrected_pvalues, n_sig_per_pair_df = compute_pvalues_for_group_pairs(real_zs_within_group,
                                                                                   null_zs_within_group,
                                                                                   df_k_real,
                                                                                   group_names,
                                                                                   n_perm,
                                                                                   THRESH,
                                                                                   MC_METHOD)

    # combine results into dataframe
    df_z = pd.DataFrame(real_zs_within_group)
    df_z.index = df_z.index.set_names("syllable")
    dunn_results_df = df_z.reset_index().melt(id_vars="syllable")

    # take the intersection of KW and Dunn's tests
    print('permutation within group-pairs (2 groups)')
    print("intersection of Kruskal-Wallis and Dunn's tests")
    intersect_sig_syllables = {}
    for pair in df_pair_corrected_pvalues.columns.tolist():
        intersect_sig_syllables[pair] = np.where((df_pair_corrected_pvalues[pair] < THRESH) & (df_k_real.is_sig))[0]
        print(pair, len(intersect_sig_syllables[pair]), intersect_sig_syllables[pair])

    return df_k_real, dunn_results_df, intersect_sig_syllables

def compute_pvalues_for_group_pairs(real_zs_within_group, null_zs, df_k_real, group_names, n_perm=10000, THRESH=0.05, MC_METHOD='fdr_bh'):
    '''

    Parameters
    ----------
    real_zs_within_group
    null_zs
    df_k_real
    group_names
    n_perm
    THRESH
    MC_METHOD

    Returns
    -------

    '''

    # do empirical p-val calculation for all group permutation
    print(f'permutation across all {len(group_names)} groups')
    print('significant syllables for each pair with FDR < ' + str(THRESH))

    p_vals_allperm = {}
    for (pair) in combinations(group_names, 2):
        p_vals_allperm[pair] = ((null_zs[pair] > real_zs_within_group[pair]).sum(0) + 1) / n_perm

    # summarize into df
    df_pval = pd.DataFrame(p_vals_allperm)

    correct_p = lambda x: multipletests(x, alpha=THRESH, method=MC_METHOD)[1]
    df_pval_corrected = df_pval.apply(correct_p, axis=1, result_type='broadcast')

    return df_pval_corrected, ((df_pval_corrected[df_k_real.is_sig] < THRESH).sum(0))


def permute_within_group_pairs(df_usage, vc, real_ranks, X_ties, N_m, group_names, rnd, n_perm):
    '''

    Parameters
    ----------
    df_usage
    vc
    real_ranks
    X_ties
    N_m
    group_names
    rnd
    n_perm

    Returns
    -------

    '''

    null_zs_within_group = {}
    real_zs_within_group = {}

    A = N_m * (N_m + 1.) / 12.

    for (i_n, j_n) in combinations(group_names, 2):

        is_i = df_usage.group == i_n
        is_j = df_usage.group == j_n

        n_mice = is_i.sum() + is_j.sum()

        ranks_perm = real_ranks[(is_i | is_j)][rnd.rand(n_perm, n_mice).argsort(-1)]
        diff = np.abs(ranks_perm[:, :is_i.sum(), :].mean(1) - ranks_perm[:, is_i.sum():, :].mean(1))
        B = (1. / vc.loc[i_n] + 1. / vc.loc[j_n])

        # also do for real data
        group_ranks = real_ranks[(is_i | is_j)]
        real_diff = np.abs(group_ranks[:is_i.sum(), :].mean(0) - group_ranks[is_i.sum():, :].mean(0))

        # add to dict
        pair = (i_n, j_n)
        null_zs_within_group[pair] = diff / np.sqrt((A - X_ties) * B)
        real_zs_within_group[pair] = real_diff / np.sqrt((A - X_ties) * B)

    return null_zs_within_group, real_zs_within_group

def run_mann_whitney_test(df, group1, group2, statistic, max_syllable, THRESH=0.05, MC_METHOD='fdr_bh'):
    '''

    Parameters
    ----------
    df
    group1
    group2
    statistic
    max_syllable
    THRESH
    MC_METHOD

    Returns
    -------

    '''

    # get mean grouped data
    grouped_data = get_session_mean_df(df, statistic, max_syllable).reset_index()

    vc = grouped_data.group.value_counts().loc[[group1, group2]]
    n_per_group = vc.values

    df_only_usage = grouped_data[range(max_syllable)]
    merged_usages_all = df_only_usage.values

    N_m, N_s = merged_usages_all.shape

    df_mw_real = pd.DataFrame([stats.mannwhitneyu(*np.array_split(merged_usages_all[:, s_i],
                                                            np.cumsum(n_per_group[:-1]))) for s_i in range(N_s)])

    exclude_sylls = run_multiple_comparisons_test(df_mw_real['pvalue'], THRESH=THRESH, MC_METHOD=MC_METHOD)

    df_mw_real['is_sig'] = df_mw_real['pvalue'] < THRESH

    df_mw_real.at[exclude_sylls, 'is_sig'] = False

    print(f"Found {df_mw_real['is_sig'].sum()} syllables that pass threshold {THRESH} with {MC_METHOD}")

    return df_mw_real, exclude_sylls

def run_ztest(df, group1, group2, statistic, max_syllable=40):
    '''

    Parameters
    ----------
    df
    group1
    group2
    statistic
    max_syllable

    Returns
    -------

    '''

    boots = bootstrap_group_means(df, group1, group2, statistic, max_syllable)

    # do a ztest on the bootstrap distributions of your 2 conditions
    pvals_ztest_boots = ztest_vect(boots[group1], boots[group2])

    significant_syllables = np.array(range(len(pvals_ztest_boots)))[pvals_ztest_boots < 0.05]

    exclude_sylls = run_multiple_comparisons_test(pvals_ztest_boots)
    syllables_to_include = [i for i in significant_syllables if i not in exclude_sylls]

    return pvals_ztest_boots, syllables_to_include

def run_ttest(df, group1, group2, statistic, max_syllable=40):
    '''

    Parameters
    ----------
    df
    group1
    group2
    statistic
    max_syllable

    Returns
    -------

    '''

    # get separated group variables
    group_stat = get_session_mean_df(df, statistic, max_syllable)

    groups = (group1, group2)
    usages = {k: group_stat.loc[k].values for k in groups}

    # run t-test
    st, p = stats.ttest_ind(usages[group1], usages[group2])

    significant_syllables = np.array(range(len(p)))[p < 0.05]

    exclude_sylls = run_multiple_comparisons_test(p)

    syllables_to_include = [i for i in significant_syllables if i not in exclude_sylls]

    return p, syllables_to_include

def run_multiple_comparisons_test(pvals_ztest_boots, THRESH=0.05, MC_METHOD='fdr_bh'):
    '''

    Parameters
    ----------
    pvals_ztest_boots

    Returns
    -------

    '''

    # significant syllables (relabeled by time used)
    exclude_sylls = np.where(multipletests(pvals_ztest_boots, alpha=THRESH, method=MC_METHOD)[0])[0]

    return exclude_sylls

def get_classifier(model_str, C=1.0, penalty='l2'):
    pass

def plot_confusion_matrix(C, y):
    '''

    Parameters
    ----------
    C
    y

    Returns
    -------

    '''

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    sns.heatmap(C, annot=True, xticklabels=np.unique(y), yticklabels=np.unique(y))

    return fig, ax

def train_classifier(df, model_str='logistic_regression', max_syllable=40, stat='usage', test_size=0.2, C=1.0, penalty='l2'):
    '''

    Parameters
    ----------
    df
    model_str
    stat
    max_syllable
    test_size
    C
    penalty

    Returns
    -------

    '''

    group_mean_df = get_session_mean_df(df, stat, max_syllable).reset_index()

    # choose input features
    X = group_mean_df[range(max_syllable)].values

    # get corresponding group labels
    y = group_mean_df.group.values

    # get train and test datasets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # replace with get_classifier
    model = LogisticRegression(multi_class='ovr', C=C, penalty=penalty)

    model.fit(x_train, y_train)

    y_score = model.predict(x_test)
    print('F1 Score = ' + str(f1_score(y_test, y_score, average='micro')))

    confusion_mat = confusion_matrix(y_test, y_score)

    fig, ax = plot_confusion_matrix(confusion_mat, y)

    return model, confusion_matrix

def normalize_matrix(C, maxiter=500):
    '''

    Parameters
    ----------
    C
    maxiter

    Returns
    -------

    '''

    A = C.copy().astype('float32')
    for i in range(maxiter):
        prev = A.copy()
        d = A.sum(0)
        d[d == 0] = 1.0
        A /= d
        d = A.sum(1)[:, None]
        d[d == 0] = 1.0
        A /= d
        if np.allclose(prev, A, atol=1e-10, rtol=1e-12):
            return A
    return A

def heldout_confusion_matrix(df, stat, max_syllable=40, C=1.0, penalty='l2'):
    '''

    Parameters
    ----------
    df
    stat
    max_syllable
    C
    penalty

    Returns
    -------

    '''

    group_mean_df = get_session_mean_df(df, stat, max_syllable).reset_index()

    # choose input features
    features = group_mean_df[range(max_syllable)].values

    # get corresponding group labels
    labels = group_mean_df.group.values

    # Build held-out classification
    true_heldout_y = []
    pred_heldout_y = []

    for ilabel in np.unique(labels):
        heldout_ind = labels != ilabel
        nonheldout_ind = labels == ilabel
        classifier = LogisticRegression(penalty=penalty,
                                        C=C,
                                        multi_class='ovr')
        classifier.fit(features[heldout_ind], labels[heldout_ind])
        pred_y = classifier.predict(features[nonheldout_ind])
        true_y = labels[nonheldout_ind]
        true_heldout_y.append(true_y)
        pred_heldout_y.append(pred_y)

    true_heldout_y = np.hstack(true_heldout_y).ravel()
    pred_heldout_y = np.hstack(pred_heldout_y).ravel()
    confusion_mat = normalize_matrix(confusion_matrix(true_heldout_y, pred_heldout_y))

    fig, ax = plot_confusion_matrix(confusion_mat, labels)

    return confusion_mat