load('forward_subset_selection_5mean_dropout_relu_F1_final3.mat')
%% Subset selection F1
figure
errorbar(mean(F1_train,2),2.7764*std(F1_train,0,2)/sqrt(5),'LineWidth',1)
hold on
errorbar(mean(F1_val,2),2.7764*std(F1_val,0,2)/sqrt(5),'LineWidth',1)
hold on
errorbar(mean(F1_test,2),2.7764*std(F1_test,0,2)/sqrt(5),'LineWidth',1)
xlim([1,12]), xlabel('Number of leads used')
ylim([0.6,0.9]), ylabel('F1')
legend('F1 on training set','F1 on validation set','F1 on testing set',...
    'Location','southeast')

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
errorbar(mean(AUC_train,2),2.2622*std(AUC_train,0,2)/sqrt(10),'LineWidth',1)
hold on
errorbar(mean(AUC_val,2),2.2622*std(AUC_val,0,2)/sqrt(10),'LineWidth',1)
hold on
errorbar(mean(AUC_test,2),2.2622*std(AUC_test,0,2)/sqrt(10),'LineWidth',1)
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
    line([0,10],[y,y],'Color','w','LineWidth',0.8,'LineStyle','--')
end
hold off

%% Heatmap test
rhythm = {"AF", "I-AVB", "LBBB", "Normal", "PAC", "PVC", "RBBB", "STD", "STE"};
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
xticklabels(rhythm)
xlabel('Prediction')
yticks(label_pos)
yticklabels(rhythm)
ylabel('Patients')
hold on
y=0;
for i=1:8
    y = y+class_size(i);
    line([0,10],[y,y],'Color','w','LineWidth',0.8,'LineStyle','--')
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
    legend(['validation AUC=',num2str(AUC_val)],['test AUC=',num2str(AUC_test)],'location','south')
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

