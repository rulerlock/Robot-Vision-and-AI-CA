% Load the data
load("result_data\matlab1113.mat");
load("result_data\batchsize512_6kepoch.mat");

% Set up x-axis values
epoch_CNN = (1:50:1661*50);
epoch_MLP = (1:50:120*50);

% Plot Accuracy
figure;
subplot(2, 1, 1);
plot(epoch_CNN, trainingMetrics.TrainingAccuracy, 'b-', 'DisplayName', 'Training Accuracy (CNN)');
hold on;
plot(epoch_MLP, trainAcc * 100, 'r--', 'DisplayName', 'Training Accuracy (MLP)');
xlabel('Epoch');
ylabel('Accuracy');
title('Training and Validation Accuracy');
legend;
grid on;
xlim([0, 2500]); % Limit x-axis to 2000 epochs

% Plot Loss
subplot(2, 1, 2);
yyaxis left
plot(epoch_CNN, trainingMetrics.TrainingLoss, 'b-', 'DisplayName', 'Training Loss (CNN)');
ylabel('Training Loss (CNN)');

yyaxis right
plot(epoch_MLP, trainLoss, 'r--', 'DisplayName', 'Training Loss (MLP)');
ylabel('Training Loss (MLP)');
xlabel('Epoch');
% ylabel('Loss');
title('Training and Validation Loss');
legend;
grid on;
xlim([0, 2500]); % Limit x-axis to 2000 epochs
