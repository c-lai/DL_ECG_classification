%% Subset selection F1
n_obs = size(F1_train,2);
ts = tinv(0.975,n_obs-1);
figure
errorbar(mean(F1_train,2),ts*std(F1_train,0,2)/sqrt(n_obs),'LineWidth',1)
hold on
errorbar(mean(F1_val,2),ts*std(F1_val,0,2)/sqrt(n_obs),'LineWidth',1)
hold on
errorbar(mean(F1_test,2),ts*std(F1_test,0,2)/sqrt(n_obs),'LineWidth',1)
xlim([1,12]), xlabel('Number of leads used')
ylim([0.65,1])
legend('training','validation','test','Location','southeast')

%% significance test
[h_norm_val_subset, p_norm_val_subset, ~] = swtest(F1_val(4,:));
[h_norm_val_full, p_norm_val_full, ~] = swtest(F1_val(12,:));
if (~h_norm_val_subset)&&(~h_norm_val_full)
    [h_val,p_val] = ttest2(F1_val(4,:),F1_val(12,:))
else
    [p_val,h_val] = ranksum(F1_val(4,:),F1_val(12,:))
end

[h_norm_test_subset, p_norm_test_subset, ~] = swtest(F1_test(4,:));
[h_norm_test_full, p_norm_test_full, ~] = swtest(F1_test(12,:));
if (~h_norm_val_subset)&&(~h_norm_val_full)
    [h_test,p_test] = ttest2(F1_test(4,:),F1_test(12,:))
else
    [p_test,h_test] = ranksum(F1_test(4,:),F1_test(12,:))
end


%% Subset selection J
figure
errorbar(mean(G_train,2),2.2622*std(G_train,0,2)/sqrt(5),'LineWidth',1)
hold on
errorbar(mean(G_val,2),2.2622*std(G_val,0,2)/sqrt(5),'LineWidth',1)
hold on
errorbar(mean(G_test,2),2.2622*std(G_test,0,2)/sqrt(5),'LineWidth',1)
xlim([1,12])
legend('J on training set','J on validation set','J on testing set',...
    'Location','southeast')

%% Subset selection AUC
figure
errorbar(mean(AUC_train,2),std(AUC_train,0,2)/sqrt(10),'LineWidth',1)
hold on
errorbar(mean(AUC_val,2),std(AUC_val,0,2)/sqrt(10),'LineWidth',1)
hold on
errorbar(mean(AUC_test,2),std(AUC_test,0,2)/sqrt(10),'LineWidth',1)
xlim([1,12])
legend('AUC on training set','AUC on validation set','AUC on testing set',...
    'Location','southeast')

%% Heatmap val
rhythm = {"AF", "I-AVB", "LBBB", "Normal", "PAC", "PVC", "RBBB", "STD", "STE"};
true = true_label_val;
pred_score = pred_score_val;
confmat = [];
class_size = zeros(1,9);
label_pos = zeros(1,9);
for c=1:9
    class_index = find(true(:,c)==1);
    class_size(c) = size(class_index, 1);
    class_pred_score = pred_score(class_index,c);
    [~, I] = sort(class_pred_score,'descend');
    class_confmat = pred_score(class_index(I),:);
    confmat = [confmat; class_confmat];
    if c==1
        label_pos(c) = class_size(c)/2;
    else
        label_pos(c) = label_pos(c-1)+class_size(c-1)/2+class_size(c)/2;
    end
end


figure
imagesc(confmat), colorbar, colormap('hot')
xticks(1:9)
xticklabels(rhythm)
xlabel('Prediction')
yticks(label_pos)
yticklabels(rhythm)
ylabel('Patients')
hold on
y=0;
for i=1:8
    y = y+class_size(i);
    line_gender([0,10],[y,y],'Color','w','LineWidth',0.8,'LineStyle','--')
end
hold off

%% Heatmap test
prediction = {"AF", "I-AVB", "LBBB", "Normal", "PAC", "PVC", "RBBB", "STD", "STE"};
% patients = {"AF(F1=0.95)", "I-AVB(F1=0.83)", "LBBB(F1=0.91)", "Normal(F1=0.70)", ...
%     "PAC(F1=0.63)", "PVC(F1=0.77)", "RBBB(F1=0.92)", "STD(F1=0.75)", "STE(F1=0.49)"};
patients = {"AF", "I-AVB", "LBBB", "Normal", "PAC", "PVC", "RBBB", "STD", "STE"};
true = true_label_test;
pred_score = pred_score_test;
confmat = [];
class_size = zeros(1,9);
label_pos = zeros(1,9);
for c=1:9
    class_index = find(true(:,c)==1);
    class_size(c) = size(class_index, 1);
    class_pred_score = pred_score(class_index,c);
    [~, I] = sort(class_pred_score,'descend');
    class_confmat = pred_score(class_index(I),:);
    confmat = [confmat; class_confmat];
    if c==1
        label_pos(c) = class_size(c)/2;
    else
        label_pos(c) = label_pos(c-1)+class_size(c-1)/2+class_size(c)/2;
    end
end


figure
imagesc(confmat), colorbar, colormap('hot')
xticks(1:9)
xticklabels(prediction)
xlabel('Interpretation')
yticks(label_pos)
yticklabels(patients)
ylabel('Patients')
hold on
y=0;
for i=1:8
    y = y+class_size(i);
    line_gender([0,10],[y,y],'Color','w','LineWidth',0.8,'LineStyle','--')
end
hold off

%% Importance polar map
figure
theta = 0:pi/6:2*pi;
rhythm = {"AF", "I-AVB", "LBBB", "Normal", "PAC", "PVC", "RBBB", "STD", "STE"};
for c = 1:9
    importance = [lead_importance(c,:), lead_importance(c,1)];
    subplot(3,3,c)
    polarplot(theta, importance, 'Marker', 'o', 'LineWidth', 1)
    thetaticks(0:30:330)
    thetaticklabels({'I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'})
    title(rhythm(c))
end

%% Permutation importance polar map
figure
theta = 0:pi/6:2*pi;
rhythm = {"AF", "I-AVB", "LBBB", "Normal", "PAC", "PVC", "RBBB", "STD", "STE"};
for c = 1:9
    importance = [lead_permu_importance(c,:), lead_permu_importance(c,1)];
    subplot(3,3,c)
    polarplot(theta, importance, 'Marker', 'o', 'LineWidth', 1)
    thetaticks(0:30:330)
    thetaticklabels({'I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'})
    title(rhythm(c))
end

%% AUC
figure
rhythm = {"AF", "I-AVB", "LBBB", "Normal", "PAC", "PVC", "RBBB", "STD", "STE"};
for c = 1:9
    subplot(3,3,c)
    [X_val,Y_val,T_val,AUC_val] = perfcurve(true_label_val(:,c),pred_score_val(:,c),1);
    plot(X_val,Y_val,'LineWidth',2), hold on, axis square
    [X_test,Y_test,T_test,AUC_test] = perfcurve(true_label_test(:,c),pred_score_test(:,c),1);
    plot(X_test,Y_test,'LineWidth',2), axis square
    title(rhythm(c))
%     legend('validation','test','location','south')
    xlabel('1 - Specificity')
    ylabel('Sensitivity')
    legend(['AUC=',num2str(round(AUC_val, 3))],['AUC=',num2str(round(AUC_test, 3))],'location','south')
end

%% concurrent
label_matrix = zeros(9);
multilabel = find(sum(true_label_train, 2)==2);
for i = 1:length(multilabel)
    idx = find(true_label_train(multilabel(i), :)==1);
    label_matrix(idx(1),idx(2))=...
        label_matrix(idx(1),idx(2))+1;
end


%% NN vs RF bar plot
F1_class_val_NN = zeros(5,9);
F1_class_train_NN = zeros(5,9);
for i = 1:5
    load(['.\result\decision_result_NN_',num2str(i)]);
    pred_label = pred_label_val;
    true = true_label_val;
    for c = 1:9
        true_c = true(:,c);
        pred_c = int32(pred_label(:,c));
        confmat = confusionmat(true_c, pred_c);
        F1 = 2*confmat(2,2)/(2*confmat(2,2)+confmat(1,2)+confmat(2,1));
        F1_class_val_NN(i, c) = F1;
    end
    pred_label = pred_label_train;
    true = true_label_train;
    for c = 1:9
        true_c = true(:,c);
        pred_c = int32(pred_label(:,c));
        confmat = confusionmat(true_c, pred_c);
        F1 = 2*confmat(2,2)/(2*confmat(2,2)+confmat(1,2)+confmat(2,1));
        F1_class_train_NN(i, c) = F1;
    end
end

F1_class_val_tree = zeros(5,9);
F1_class_train_tree = zeros(5,9);
for i = 1:5
    load(['.\result\decision_result_tree_1000_subset_',num2str(i)]);
    pred_label = pred_label_val;
    true = true_label_val;
    for c = 1:9
        true_c = true(:,c);
        pred_c = int32(pred_label(:,c));
        confmat = confusionmat(true_c, pred_c);
        F1 = 2*confmat(2,2)/(2*confmat(2,2)+confmat(1,2)+confmat(2,1));
        F1_class_val_tree(i, c) = F1;
    end
    pred_label = pred_label_train;
    true = true_label_train;
    for c = 1:9
        true_c = true(:,c);
        pred_c = int32(pred_label(:,c));
        confmat = confusionmat(true_c, pred_c);
        F1 = 2*confmat(2,2)/(2*confmat(2,2)+confmat(1,2)+confmat(2,1));
        F1_class_train_tree(i, c) = F1;
    end
end

decision_val = zeros(1,9);
p_value_val = zeros(1,9);
for c = 1:9
    [h,p] = ttest2(F1_class_val_tree(:,c),F1_class_val_NN(:,c));
    decision_val(c)=h;
    p_value_val(c)=p;
end

% figure
% subplot(2,1,1)
% hold on
% c = 1:9;
% bar(c-0.2,mean(F1_class_val_NN,1),0.3);
% bar(c+0.2,mean(F1_class_val_tree,1),0.3);
% errorbar(c-0.2,mean(F1_class_val_NN,1),2.7764*std(F1_class_val_NN,0,1)/sqrt(5),...
%     'LineStyle','none','LineWidth',0.8,'Color','k')
% errorbar(c+0.2,mean(F1_class_val_tree,1),2.7764*std(F1_class_val_tree,0,1)/sqrt(5),...
%     'LineStyle','none','LineWidth',0.8,'Color','k')
% xlim([0.1,9.9]),ylim([0,1])
% box on
% hold off
% 
% subplot(2,1,2)
% hold on
% c = 1:9;
% bar(c-0.2,mean(F1_class_train_NN,1),0.3);
% bar(c+0.2,mean(F1_class_train_tree,1),0.3);
% errorbar(c-0.2,mean(F1_class_train_NN,1),2.7764*std(F1_class_train_NN,0,1)/sqrt(5),...
%     'LineStyle','none','LineWidth',0.8,'Color','k')
% errorbar(c+0.2,mean(F1_class_train_tree,1),2.7764*std(F1_class_train_tree,0,1)/sqrt(5),...
%     'LineStyle','none','LineWidth',0.8,'Color','k')
% xlim([0.1,9.9]),ylim([0,1])
% set(gca, 'YDir', 'reverse')
% box on
% hold off

figure
subplot(1,2,1)
hold on
c = 1:9;
barh(c-0.2,mean(F1_class_train_NN,1),0.3);
barh(c+0.2,mean(F1_class_train_tree,1),0.3);
ylim([0.1,9.9]),xlim([0,1])
set(gca, 'XDir', 'reverse')
set(gca, 'yDir', 'reverse')
set(gca, 'yticklabel', [])
box on
hold off

subplot(1,2,2)
hold on
c = 1:9;
barh(c-0.2,mean(F1_class_val_NN,1),0.3);
barh(c+0.2,mean(F1_class_val_tree,1),0.3);
ylim([0.1,9.9]),xlim([0,1])
set(gca, 'yDir', 'reverse')
box on
legend('neural network', 'random forest', 'location','southeast')
hold off

%% F1
F1_class_test_NN = zeros(1,9);

pred_label = pred_label_test;
true = true_label_test;
for c = 1:5
    true_c = int32(true(:,c));
    pred_c = int32(pred_label(:,c));
    confmat = confusionmat(true_c, pred_c);
    F1 = 2*confmat(2,2)/(2*confmat(2,2)+confmat(1,2)+confmat(2,1));
    F1_class_test_NN(c) = F1;
end

%% AUC new data
figure
rhythm = {"AF", "AVB", "BBB", "Normal", "ST Abnormality"};
for c = 1:5
    subplot(3,3,c)
    [X_test,Y_test,T_test,AUC_test] = perfcurve(true_label_test(:,c),pred_score_test(:,c),1);
    plot(X_test,Y_test,'LineWidth',2), axis square
    title(rhythm(c))
    xlabel('1 - Specificity')
    ylabel('Sensitivity')
    legend(['test AUC=',num2str(AUC_test)],'location','south')
end

%% get rid of PAV, PVC
PAC_idx = find(true_label_test(:,5)==1);
PVC_idx = find(true_label_test(:,6)==1);

true_label_test([PAC_idx;PVC_idx], :)=[];
pred_label_test([PAC_idx;PVC_idx], :)=[];
pred_score_test([PAC_idx;PVC_idx], :)=[];

true_label_test(:,[5,6])=[];
pred_label_test(:,[5,6])=[];
pred_score_test(:,[5,6])=[];

%%
complete_f1 = [0.5361, 0.5365, 0.5360, 0.5356, 0.5369, 0.5392, 0.5356, 0.5377, 0.5364, 0.5350];
subset_f1 = [0.5478, 0.5477, 0.5481, 0.5463, 0.5472, 0.5475, 0.5482, 0.5468, 0.5464, 0.5461];
[h_test,p_test] = ttest2(complete_f1,subset_f1)

%% gender statistics
hea_files = dir('./*.hea');
male = 0;
female = 0;
for i = 1:size(hea_files)
    fid = fopen(hea_files(i).name);
    for l = 1:15
        line_gender = fgetl(fid);
    end
    line_diagnosis = fgetl(fid);
    fclose(fid);
    gender = line_gender(7:end);
    if ~(contains(line_diagnosis,'284470004')||contains(line_diagnosis,'59118001'))
        if length(gender) == 4
            male=male+1;
        elseif length(gender) == 6
            female=female+1;
        end
    end 
end
        
    
    
