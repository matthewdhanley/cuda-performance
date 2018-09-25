clear all; close all;

filenames = ["cpu_add_times.txt","gpu_add_times.txt"];

% read in the file
fig = figure;
for i=1:length(filenames)
    disp(filenames(i))
    f = fopen(filenames(i));
    data = csvread(filenames(i));
    loglog(data(:,1),data(:,2),'*-');
    hold on;
    grid on;
end

