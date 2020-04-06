%% load data
classes = {'AF','I-AVB','LBBB','Normal','PAC','PVC','RBBB','STD','STE'};
output_directory = '.\result\lead2_ResNet24';
input_directory = '.\Training_WFDB\*\*.txt';

output_files = {};
labels_pred = [];
labels_true = [];
for f = dir(output_directory)'
    if exist(fullfile(output_directory, f.name), 'file') == 2 && f.name(1) ~= '.' && all(f.name(end - 2 : end) == 'csv')
        output_files{end + 1} = f.name;
        
        prediction = readmatrix(fullfile(output_directory, f.name));
        label_pred = find(prediction(1,:));
        labels_pred = [labels_pred, label_pred];
    end
end

for f = dir(input_directory)'
    if exist(fullfile(f.folder, f.name), 'file') == 2 && f.name(1) ~= '.' && all(f.name(end - 2 : end) == 'txt')
        label_true = fileread(fullfile(f.folder, f.name));
        label_true = convertCharsToStrings(label_true);
        labels_true = [labels_true, label_true];
    end
end

figure, histogram(labels_pred)

%% calculate accuracy
matches = [];
for i = 1:length(labels_pred)
    label_pred = classes{labels_pred(i)};
    label_true = labels_true(i);
    match = contains(label_true, label_pred);
    matches = [matches, match];
end
accuracy = mean(matches)

%% true label distribution
num_multiclass = 0;
labels_true_val = [];
for i = 1:length(labels_true)
    label_true = labels_true(i);
    if contains(label_true, ',')
        label_true = split(label_true,',');
        num_multiclass = num_multiclass+1;
        label_true_val = find(contains(classes, label_true(1)));%???
    else
        label_true_val = find(contains(classes, label_true));
    end
    labels_true_val = [labels_true_val, label_true_val];
end

figure, histogram(labels_true_val)
        