clear all; close all;

filenames_cpu = ["cpu_add_times_100_10000.txt"
                 "cpu_add_times_1000_10000.txt"
                 "cpu_add_times_10000_10000.txt"
                 "cpu_add_times_100000_10000.txt"];

filenames_gpu = ["gpu_add_times_100_10000.txt"
                 "gpu_add_times_1000_10000.txt"
                 "gpu_add_times_10000_10000.txt"
                 "gpu_add_times_100000_10000.txt"];

plot_perf(filenames_cpu);
plot_perf(filenames_gpu);

function fig = plot_perf(filenames)
    % read in the file
    fig = figure;
    for i=1:length(filenames)
        disp(filenames(i))
        f = fopen(filenames(i));
        data = fscanf(f, '%f');
        fclose(f);
        plot(data,'o','color',rand(1,3));
        hold on;
    end
end

