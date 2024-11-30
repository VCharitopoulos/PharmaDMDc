clc;
clear all;
close all;

% Folder
results_folder = 'MOESP_results';
if ~exist(results_folder, 'dir')
    mkdir(results_folder);
end

% Load data
data = csvread('datasets.csv', 1, 0);  

% Extract estimation data
StateData = data(:, [4, 5, 6, 7]);  
InputData = data(:, [1, 2, 3]);    
first_value_T = StateData(1, :); 

% System identification using MOESP
opt = n4sidOptions('N4weight','MOESP');
sys = n4sid(iddata(StateData, InputData), 4, 'Form', 'canonical', 'DisturbanceModel', 'none', opt);

% Eigenvalues
[V, D] = eig(sys.A);
%disp('Eigenvalues:');
disp(D);

% Save state-space matrices to CSV
State_Space = [sys.A, sys.B; sys.C, sys.D];
headers = {'A', 'B', 'C', 'D'};
state_space_file = fullfile(results_folder, 'MOESP_MATLAB_TSG.csv');
fileID = fopen(state_space_file, 'w');
fprintf(fileID, '%s,', headers{1:end-1});
fprintf(fileID, '%s\n', headers{end});
fclose(fileID);
dlmwrite(state_space_file, State_Space, '-append');

% Validation datasets 
validation_indices = {
    {8:10, 11:14}, % Validation 1
    {15:17, 18:21}, % Validation 2
    {22:24, 25:28}  % Validation 3
};

% Process each dataset
state_labels = {'D5', 'D10', 'D50', 'D90'};
for v = 1:length(validation_indices)
    input_indices = validation_indices{v}{1};
    state_indices = validation_indices{v}{2};

    InputData_v = data(:, input_indices);
    StateData_v = data(:, state_indices);
    X0_validation = StateData_v(1, :);

    t = (1:length(StateData_v))';  
    outputs_validation = lsim(sys, InputData_v', t, X0_validation);

    % Error metrics
    metrics_validation = calculateMetrics(StateData_v, outputs_validation);
    R2_validation = metrics_validation.R2;
    MSE_validation = metrics_validation.MSE;
    MRE_validation = metrics_validation.MRE;

    fprintf('Validation %d Data Metrics:\n', v);
    fprintf('R-squared (R²): %.4f\n', R2_validation);
    fprintf('Mean Squared Error (MSE): %.4f\n', MSE_validation);
    fprintf('Mean Relative Error (MRE): %.4f%%\n', MRE_validation);

    % Plots
    for i = 1:length(state_indices)
        figure;
        plot(StateData_v(:, i), 'b', 'DisplayName', sprintf('%s Simulation data (Validation %d)', state_labels{i}, v), 'LineWidth', 1);
        hold on;
        plot(outputs_validation(:, i), 'r--', 'DisplayName', sprintf('%s Predicted (MATLAB n4sid)', state_labels{i}), 'LineWidth', 1);
        xlabel('Time (s)', 'FontSize', 6);
        ylabel(sprintf('%s (\\mum)', state_labels{i}), 'FontSize', 6);
        legend('Location', 'southoutside', 'FontSize', 5, 'Orientation', 'horizontal', 'Box', 'off');
        title(sprintf('%s Validation %d vs Predicted Output', state_labels{i}, v), ...
            {sprintf('R²: %.4f', R2_validation(i)), sprintf('MSE: %.4f', MSE_validation(i)), sprintf('MRE: %.4f%%', MRE_validation(i))}, 'FontSize', 6);
        grid off;
        set(gca, 'LooseInset', max(get(gca, 'TightInset'), 0.02)); 
        filename_png = fullfile(results_folder, sprintf('Validation%d_%s.png', v, state_labels{i}));
        exportgraphics(gcf, filename_png, 'Resolution', 300); 
    end

    % CSV file
    validation_results_t0 = [StateData_v, outputs_validation];
    validation_file = fullfile(results_folder, sprintf('Validation%d_MOESP_TSG.csv', v));
    fileID = fopen(validation_file, 'w');
    fprintf(fileID, '%s,', headers{1:end-1});
    fprintf(fileID, '%s\n', headers{end});
    fclose(fileID);
    dlmwrite(validation_file, validation_results_t0, '-append');
end

function metrics = calculateMetrics(original, predicted)
    R2 = 1 - sum((original - predicted).^2) ./ sum((original - mean(original)).^2);
    MSE = mean((original - predicted).^2);
    MRE = mean(abs(original - predicted) ./ original) * 100;   
    metrics.R2 = R2;
    metrics.MSE = MSE;
    metrics.MRE = MRE;
end
