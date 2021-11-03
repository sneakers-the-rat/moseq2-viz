'''
Functions for creating fingerprint plots and linear classifier
'''
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from sklearn.model_selection import StratifiedKFold, LeaveOneOut, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.base import clone

import pandas as pd
import numpy as np


def create_fingerprint_dataframe(scalar_df, mean_df, bin_num=None, groupby_list=['group', 'uuid']):
    '''
    create fingerprint dataframe from scalar_df and mean_df

    Args:
        scalar_df ([pandas.DataFrame]): scalar summary dataframe generated from scalars_to_dataframe
        mean_df ([pandas.DataFrame]): syllable mean dataframe from compute_behavioral_statistics
        bin_num ([int], optional): number of bins for the features. Defaults to None.
        groupby_list (list, optional): the list of levels the fingerprint dataframe should be grouped by. Defaults to ['group', 'uuid'].

    Returns:
        summary ([pandas.DataFrame]): fingerprint dataframe
        range_dict ([dict]): dictionary that hold min max values of the features
    '''
    # process mean_df
    mean_df = mean_df[['group', 'uuid', 'syllable', 'usage']] # pandas version 1.0.5 doesn't support a list of index, therefore subset mean_df
    mean_df.set_index(['group', 'uuid'], inplace = True)
    syll_summary = mean_df.pivot(index= mean_df.index, columns='syllable')
    max_syll_id = len(syll_summary.columns)-1
    syll_summary = pd.DataFrame({'MoSeq': syll_summary.values.tolist()}, index = syll_summary.index)
    
    # process scalar_df
    # Plotting position to center, velocity, height, length
    upper_height = np.ceil(scalar_df['height_ave_mm'].max())
    upper_length = np.ceil(scalar_df['length_mm'].max())
    # to account for the super high speed mice we use 99 percentile
    upper_velocity = np.ceil(np.percentile(np.abs(scalar_df['velocity_2d_mm'].dropna()), 99)*30) # to account for 30fps we multiply 30
    upper_position = scalar_df['dist_to_center_px'].max()
    
    # use total number of syllables 
    if bin_num is None:
        bin_num = max_syll_id+1 # num of bins (default to match the total number of syllables)
    
    # Functions to be applied on the groups
    # If we want to give the users freedom to plot anything they want, then we need to preset all the apply fn.
    heightfn = lambda df: np.histogram(df['height_ave_mm'], np.linspace(0, upper_height, bin_num))[0]
    lengthfn = lambda df: np.histogram(df['length_mm'], np.linspace(20, upper_length, bin_num))[0] # set mouse mimimum length to 20 per Alex's drug paper
    # Multiply the velocity by frame rate because it is only diff distance between frames
    # what is a good cutoff speed for velocity, right now max is 24*30
    velocityfn = lambda df: np.histogram(np.abs(30*df['velocity_2d_mm']), np.linspace(0, upper_velocity, bin_num))[0]
    positionfn = lambda df: np.histogram(df['dist_to_center_px']/upper_position, np.linspace(0, 1, bin_num))[0]
    

    scalar_df = scalar_df.groupby(groupby_list)
    scalar_summary = pd.concat([scalar_df.apply(heightfn), scalar_df.apply(lengthfn), scalar_df.apply(velocityfn), scalar_df.apply(positionfn)], axis=1, join="inner")
    scalar_summary.columns = ['Height', 'Length', 'Speed', 'Position']
    
    
    # join the two data frames
    summary = pd.concat([scalar_summary, syll_summary], axis=1, join='inner')
    print('Grouping/level in the summary is', summary.index.names)
    [1, 20, upper_height, upper_length, max_syll_id]
    range_dict = {'Height': (0, upper_height), 'Length': (20, upper_length), 'Speed':(0, upper_velocity), 'Position': (0,1), 'MoSeq':(0, max_syll_id)}
    return summary, range_dict


def plotting_fingerprint(summary, range_dict, scalar=None, num_level = 1, level_names = ['Group'], vmin=0, vmax = 0.2):
    '''
    plot the fingerprint heatmap

    Args:
        summary ([pandas.DataFrame]): fingerprint dataframe
        range_dict ([dict]): dictionary that hold min max values of the features
        scalar ([sklearn.preprocessing scalar object], optional): Scalar for scaling the data by session. Defaults to None.
        num_level (int, optional): the number of groupby levels. Defaults to 1.
        level_names (list, optional): list of names of the levels. Defaults to ['Group'].
        vmin (int, optional): min value the figure color map covers. Defaults to 0.
        vmax (float, optional): max value the figure color map covers. Defaults to 0.2.

    Raises:
        Exception: num_levels greater than the existing levels
    '''
    # ensure number of groups is not over the number of available levels
    if num_level > len(summary.index.names):
        raise Exception('Too many levels to unpack. num_level should be less than', len(summary.index.names))
    
    levels = []
    level_plot = []
    level_ticks = []
    for i in range(num_level):
        level = summary.index.get_level_values(i)
        level_label = LabelEncoder().fit_transform(level)
        find_mid = (np.diff(np.r_[0,np.argwhere(np.diff(level_label)).ravel(),len(level_label)])/2).astype('int32')
        # store level value
        levels.append(level)
        level_plot.append(level_label)
        level_ticks.append(np.r_[0,np.argwhere(np.diff(level_label)).ravel()] + find_mid)
    
    # col_num = number of grouping/level + column in summary
    col_num = num_level + len(summary.columns)

    # https://matplotlib.org/stable/tutorials/intermediate/gridspec.html
    fig = plt.figure(1, figsize=(20, 18), facecolor='white')

    gs = GridSpec(2, col_num, wspace=0.1, hspace=0.1,
              width_ratios=[1]*num_level+[8]*(col_num-num_level), height_ratios = [10,0.1], figure=fig)

    # plot the level(s)
    for i in range(num_level):
        temp_ax = fig.add_subplot(gs[0,i])
        temp_ax.set_title(level_names[i], fontsize=20)
        temp_ax.imshow(level_plot[i][:,np.newaxis], aspect = 'auto', cmap = 'Set3')
        plt.yticks(level_ticks[i], levels[i][level_ticks[i]], fontsize=20)
        
        temp_ax.get_xaxis().set_ticks([])
    
    # plot the data
    plot_col = ['Position', 'Speed', 'Height', 'Length', 'MoSeq']
    xlabel = ['Normalized Distance to Center', 'Speed\n(mm/s)', 'Height\n(mm)', 'Length\n(mm)', 'Syllable ID']
    for i in range(col_num-num_level):
        name = plot_col[i]
        temp_ax = fig.add_subplot(gs[0,i+num_level])
        temp_ax.set_title(name, fontsize=20)
        data = np.stack(summary[plot_col[i]])
        data = data/data.sum(1, keepdims=True)
        if scalar:
            data = scalar.fit_transform(data.T).T
        # top to bottom is 0-20 for y axis
        pc = temp_ax.imshow(data, aspect = 'auto', interpolation = 'nearest', vmin = vmin, vmax = vmax, extent=[range_dict[name][0], range_dict[name][1], len(summary), 0])
        temp_ax.set_xlabel(xlabel[i], fontsize=10)
        # https://stackoverflow.com/questions/14908576/how-to-remove-frame-from-matplotlib-pyplot-figure-vs-matplotlib-figure-frame
        temp_ax.get_yaxis().set_ticks([])
        temp_ax.axis = 'tight'
    
    # plot colorbar
    cb = fig.add_subplot(gs[1,-1])
    plt.colorbar(pc, cax=cb, orientation='horizontal')

    # specify labels for feature scaling
    if scalar:
        cb.set_xlabel('Min Max')
    else:
        cb.set_xlabel('Percentage Usage')


def classifier_fingerprint(summary, features = 'MoSeq', scalar = None, scalar_list = ['Height', 'Length', 'Speed', 'Position'], target = ['group'], param_search = True, C_list= np.logspace(-6,3, 50), model_type = 'lr', cv='loov', n_splits = 5):
    '''
    classifier using the fingerprint dataframe

    Args:
        summary ([pandas.DataFrame]): fingerprint dataframe
        features (str, optional): Features for the classifier. 'MoSeq' for MoSeq syllables or 'Scalar' for MoSeq scalar values. Defaults to 'MoSeq'.
        scalar ([sklearn.preprocessing scalar object], optional): Scalar for scaling the data by feature. Defaults to None.
        scalar_list (list, optional): list of MoSeq scalar values as classifier features. Defaults to ['Height', 'Length', 'Speed', 'Position'].
        target (list, optional): labels the classifier predicts. Defaults to ['group'].
        param_search (bool, optional): run GridSearchCV to find the regularization param for classifier. Defaults to True.
        C_list ([type], optional): list of C regularization paramters to search through. Defaults to np.logspace(-6,3, 50).
        model_type (str, optional): name of the linear classifier. 'lr' for logistic regression or 'svc' for linearSVC. Defaults to 'lr'.
        cv (str, optional): cross validation type. 'loov' for LeaveOneOut 'skf' for StratifiedKFold. Defaults to 'loov'.
        n_splits (int, optional): number of splits for StratifiedKFold. Defaults to 5.

    Returns:
        y_true ([np.array]): array for true label
        y_pred ([np.array]): array for predicted label
        real_f1 ([np.array]): array for f1 score
        true_coef ([np.array]): array for model weights
        y_shuffle_true ([np.array]): array for shffuled label
        y_shuffle_pred ([np.array]): array for shuffled predicted label
        shuffle_f1 ([np.array]): array for shuffled f1 score
        shuffle_coef ([np.array]): array for shuffled model weights
    '''
    # set up data for classifier
    if features == 'MoSeq':
        X = np.stack(summary['MoSeq'])
    else:    
        X = np.stack([np.hstack(row) for row in summary[scalar_list].values])
        
    print(X.shape)
    X = X/X.sum(1, keepdims=True) # normalize data
    summary = summary.reset_index()
    y = summary[target].squeeze()

    # set up model type
    if model_type == 'lr':
        clf = LogisticRegression(multi_class='ovr')
    else:
        clf = LinearSVC(multi_class='ovr')

    if param_search:
        parameters = {'C':C_list}
        grid_search = GridSearchCV(clf, parameters, scoring='accuracy')
        grid_search.fit(X,y)
        # set the best parameter for the classifier
        clf = clf.set_params(**grid_search.best_params_)
        print('classifier', clf)

    if cv == 'loov':
        cv = LeaveOneOut()
    else:
        cv = StratifiedKFold(n_splits=n_splits)
    
    y_true = []
    y_pred = []
    true_coef = []
    real_f1 = []
    y_shuffle_true = []
    y_shuffle_pred = []
    shuffle_coef = []
    shuffle_f1 = []

    for i in range(100):
        # shuffle y for shuffle analysis
        y_shuffle = np.random.permutation(y)
        
        for train_ix, test_ix in cv.split(X,y):
            # split data
            X_train, X_test = X[train_ix, :], X[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]
            y_shuffle_train, y_shuffle_test = y_shuffle[train_ix], y_shuffle[test_ix]
            # feature scalar
            if scalar:
                X_train = scalar.fit_transform(X_train)
                X_test = scalar.transform(X_test)

            # fit model on real data
            clf = clone(clf) # reset model to prevent data leakage
            clf.fit(X_train, y_train)
            # evaluate model
            y_true.append(y_test)
            y_hat = clf.predict(X_test)
            y_pred.append(y_hat)
            real_f1.append(f1_score(y_test, y_hat, average='macro'))
            true_coef.append(clf.coef_[0])

            # shuffle data
            clf=clone(clf)
            clf.fit(X_train, y_shuffle_train)
            # evaluate 
            y_shuffle_true.append(y_shuffle_test)
            y_hat = clf.predict(X_test)
            y_shuffle_pred.append(y_hat)
            shuffle_f1.append(f1_score(y_shuffle_test, y_hat, average='macro'))
            shuffle_coef.append(clf.coef_[0])

    return np.hstack(y_true), np.hstack(y_pred), np.array(real_f1), np.array(true_coef), np.hstack(y_shuffle_true), np.hstack(y_shuffle_pred), np.array(shuffle_f1), np.array(shuffle_coef)

def plot_cm(y_true, y_pred, y_shuffle_true, y_shuffle_pred):
    '''
    plot confusion matrix

    Args:
        y_true ([np.array]): array for true label
        y_pred ([np.array]): array for predicted label
        y_shuffle_true ([np.array]): array for shffuled label
        y_shuffle_pred ([np.array]): array for shuffled predicted label
    '''
    fig = plt.figure(figsize=(23, 10), facecolor='white')
    gs = GridSpec(ncols=3, nrows=1, wspace=0.1, figure = fig, width_ratios=[10,10,0.3])
    fig_ax = fig.add_subplot(gs[0,0])
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = cm/cm.sum(1, keepdims=True)
    pc = fig_ax.imshow(cm, cmap="Blues", vmin = 0, vmax =1)
    fig_ax.set_title(f'Real Accuracy {np.round(accuracy_score(y_true, y_pred),2)}')
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    fig_ax.set_xlabel('Predicted')

    fig_ax = fig.add_subplot(gs[0,1])
    cm = confusion_matrix(y_shuffle_true, y_shuffle_pred, labels=labels)
    cm = cm/cm.sum(1, keepdims=True)
    fig_ax.imshow(cm, cmap="Blues", vmin = 0, vmax =1)
    fig_ax.set_title(f'Shuffle Accuracy {np.round(accuracy_score(y_shuffle_true, y_shuffle_pred),2)}')
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    fig_ax.set_xlabel('Predicted')

    # plot colorbar
    cb = fig.add_subplot(gs[0,2])
    plt.colorbar(pc, cax=cb, orientation='vertical')
    cb.set_xlabel('')
    plt.show()