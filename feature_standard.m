%%
col_mean = mean(features_bm_2_train, 1);
col_std = std(features_bm_2_train, 1)+0.001;

features_bm_2_train_standard = (features_bm_2_train - repmat(col_mean, size(features_bm_2_train, 1), 1))...
    ./repmat(col_std, size(features_bm_2_train, 1), 1);
features_bm_2_dev_standard = (features_bm_2_dev - repmat(col_mean, size(features_bm_2_dev, 1), 1))...
    ./repmat(col_std, size(features_bm_2_dev, 1), 1);
%%
col_mean = mean(features_bm_2_all, 1);
col_std = std(features_bm_2_all, 1)+0.001;

features_bm_2_all_standard = (features_bm_2_all - repmat(col_mean, size(features_bm_2_all, 1), 1))...
    ./repmat(col_std, size(features_bm_2_all, 1), 1);
