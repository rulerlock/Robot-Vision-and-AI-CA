% Load the data
load("result_data\matlab1113.mat");


% Set up x-axis values
epoch_d1 = (1:50:epochs);


% Plot Accuracy
figure;
subplot(2, 1, 1);
plot(epoch_d1, trainingMetrics.TrainingAccuracy, 'b-', 'DisplayName', 'Training Accuracy (CNN)');
hold on;
plot(epoch_d2, trainAcc * 100, 'r--', 'DisplayName', 'Training Accuracy (MLP)');
xlabel('Epoch');
ylabel('Accuracy');
title('Training and Validation Accuracy');
legend;
grid on;
xlim([0, 2500]); % Limit x-axis to 2000 epochs

% Plot Loss
subplot(2, 1, 2);
yyaxis left
plot(epoch_d1, trainingMetrics.TrainingLoss, 'b-', 'DisplayName', 'Training Loss (CNN)');
ylabel('Training Loss (CNN)');

yyaxis right
plot(epoch_d2, trainLoss, 'r--', 'DisplayName', 'Training Loss (MLP)');
ylabel('Training Loss');
xlabel('Epoch');
% ylabel('Loss');
title('Training and Validation Loss');
legend;
grid on;
xlim([0, 2500]); % Limit x-axis to 2000 epochs

load("result_data\batchsize512_6kepoch.mat");

epoch_d2 = (1:50:epochs);
