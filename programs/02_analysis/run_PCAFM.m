% Location of this script
this_file = mfilename('fullpath');
this_dir  = fileparts(this_file);

% Go up explicitly
programs_dir = fileparts(this_dir);
project_root = fileparts(programs_dir);

% Build paths
ff_portfolios_path = fullfile(project_root, 'data', 'derived', 'gx_inputs', 'ff');
jkp_portfolios_path = fullfile(project_root, 'data', 'derived', 'gx_inputs', 'jkp');
controls_path       = fullfile(project_root, 'data', 'derived', 'gx_inputs', 'controls');
save_path = fullfile(project_root, 'data', 'derived', 'gx_fmb');

% Define prefixes for VW and EW
prefixes = {'vw_','ew_'};

% Loop through VW and EW
for p = 1:length(prefixes)
    prefix = prefixes{p};

    % Read in FMB control variables
    control_files = dir(fullfile(controls_path, [prefix, '*.csv']));
    [controls_m, control_var_names] = read_controls(control_files);
    controls_m = controls_m*100;

    % Set paths to read in FF and JKP test portfolios
    ff_files = dir(fullfile(ff_portfolios_path, [prefix, '*.csv']));
    jkp_files = dir(fullfile(jkp_portfolios_path, [prefix, '*.csv']));


    % ------------------------- Third Version: FF + JKP Combined Test Portfolios -------------------------
    if ~isempty(ff_files) && ~isempty(jkp_files)
        % Read in FF test portfolios
        ff_combined = combine_files(ff_files);

        % Read in JKP test portfolios 
        jkp_file_path = fullfile(jkp_files(1).folder, jkp_files(1).name);
        testport_jkp = readtable(jkp_file_path, 'ReadRowNames', true, 'VariableNamingRule', 'preserve');

        testport_jkp_m = table2array(testport_jkp);
        testport_jkp_m = testport_jkp_m*100;

        % Combine the FF and JKP matrices
        testport_combined = [ff_combined, testport_jkp_m]; 

        % Run PCAFM
        results = PCAFM(testport_combined', controls_m', 20, floor(0.75 * size(controls_m, 1)^(1/3)), control_var_names);
        save_results(results, save_path, 'FF_JKP', prefix);
    end

end

disp('Processing complete.');

% ------------------------- Helper Functions -------------------------
function testport_combined = combine_files(file_list)
    testport_tables = {};
    for file_idx = 1:length(file_list)
        file_path = fullfile(file_list(file_idx).folder, file_list(file_idx).name);
        testport = readtable(file_path, 'ReadRowNames', true, 'VariableNamingRule', 'preserve');
        testport_tables{end+1} = testport;
    end
    testport_combined = testport_tables{1};
    for i = 2:length(testport_tables)
        testport_combined = [testport_combined, testport_tables{i}];
    end
    testport_combined = table2array(testport_combined);
end

function [controls_m, control_var_names] = read_controls(file_list)
    control_file_path = fullfile(file_list(1).folder, file_list(1).name);
    controls = readtable(control_file_path, 'ReadRowNames', true, 'VariableNamingRule', 'preserve');
    controls_m = table2array(controls);
    control_var_names = controls.Properties.VariableNames;
end

function save_results(results, save_path, label, prefix)

    % Ensure folder exists
    if ~exist(save_path, 'dir')
        [ok,msg] = mkdir(save_path);
        assert(ok, "Could not create folder:\n%s\n%s", save_path, msg);
    end

    % Ensure folder is writable
    [fid, errmsg] = fopen(fullfile(save_path, '.write_test.tmp'), 'w');
    assert(fid ~= -1, "Folder not writable:\n%s\n%s", save_path, errmsg);
    fclose(fid);
    delete(fullfile(save_path, '.write_test.tmp'));

    output_filename = fullfile(save_path, ['PCAFM_results_', label, '_', prefix, '.xlsx']);
    writetable(format_results(results), output_filename, 'WriteRowNames', true);
    disp(['Results written to ', output_filename]);

    output_filename_nozero = fullfile(save_path, ['PCAFM_results_', label, '_nozero_', prefix, '.xlsx']);
    writetable(format_results(results, true), output_filename_nozero, 'WriteRowNames', true);
    disp(['No-zero results written to ', output_filename_nozero]);
end

function results_table = format_results(results, remove_zero_beta)
    if nargin < 2
        remove_zero_beta = false;
    end

    if remove_zero_beta
        % Extract table while preserving row names
        coefficients_table = results.Gammahat_nozero;
        variance_table = results.avarhat_nozero;
    else
        coefficients_table = results.Gammahat;
        variance_table = results.avarhat;
    end

    % Convert tables to arrays for calculations
    coefficients = coefficients_table{:,:};
    variance = variance_table{:,:};
    factor_names = coefficients_table.Properties.RowNames;

    % Drop "Zero-beta" row if needed
    if remove_zero_beta
        zero_beta_row_idx = strcmp(factor_names, 'Zero-beta');
        coefficients(zero_beta_row_idx, :) = [];
        factor_names(zero_beta_row_idx) = [];
    end

    % Compute statistics
    standard_error = sqrt(variance);
    t_stat = coefficients ./ standard_error;

    % Round results
    coefficients = round(coefficients, 4);
    standard_error = round(standard_error, 4);
    t_stat = round(t_stat, 4);

    column_names = coefficients_table.Properties.VariableNames;  

    % Create results table
    results_table = table('RowNames', factor_names);
    for i = 1:numel(column_names)
        col_name = column_names{i};  
        results_table.([col_name '_Beta']) = coefficients(:, i);
        results_table.([col_name '_SE']) = standard_error(:, i);
        results_table.([col_name '_tStat']) = t_stat(:, i);
    end
end
