% Location of this script
this_file = mfilename('fullpath');
this_dir  = fileparts(this_file);

% Navigate up to project root 
programs_dir = fileparts(this_dir);
project_root = fileparts(programs_dir);

% Define data paths (relative to project root)
aptree_cleaned_path = fullfile(project_root, 'data', 'derived', 'gx_inputs', 'aptree');

% Define the save path for results
save_path = fullfile(project_root, 'data', 'derived', 'gx_fmb');

% Define prefixes for VW and EW controls
control_prefixes = {'vw_', 'ew_'};

% Read test asset data (Pelger test assets)
test_assets_file = fullfile(aptree_cleaned_path, 'aptree_test_assets.csv');
test_assets_table = readtable(test_assets_file, 'ReadRowNames', true, 'VariableNamingRule', 'preserve');
test_assets_matrix = table2array(test_assets_table);
test_assets_matrix = test_assets_matrix * 100; % Scaling

% Loop through VW and EW control files
for p = 1:length(control_prefixes)
    control_prefix = control_prefixes{p};
    
    % Identify control file
    control_file = fullfile(aptree_cleaned_path, [control_prefix, 'controls_aptree.csv']);
    
    if exist(control_file, 'file')
        % Read control factors
        controls_table = readtable(control_file, 'ReadRowNames', true, 'VariableNamingRule', 'preserve');
        controls_matrix = table2array(controls_table);
        controls_matrix = controls_matrix * 100; % Scaling

        control_var_names = controls_table.Properties.VariableNames;

        % Ensure common dates between test assets and controls
        common_dates = intersect(test_assets_table.Properties.RowNames, controls_table.Properties.RowNames);
        test_assets_filtered = test_assets_table(common_dates, :);
        controls_filtered = controls_table(common_dates, :);

        % Convert to matrices
        test_assets_matrix_filtered = table2array(test_assets_filtered);
        controls_matrix_filtered = table2array(controls_filtered);

        % Run PCAFM
        results = PCAFM(test_assets_matrix_filtered', controls_matrix_filtered', 20, floor(0.75 * size(controls_matrix_filtered, 1)^(1/3)), control_var_names);

        % Save results
        save_results(results, save_path, 'aptree', control_prefix);
    else
        warning(['No control file found for ', control_prefix]);
    end
end

disp('Processing complete.');

% ------------------------- Helper Functions -------------------------
function save_results(results, save_path, label, control_prefix)
    % Adjust file naming convention to clarify prefix applies to controls
    output_filename = fullfile(save_path, ['PCAFM_results_', label, '_', control_prefix, '.xlsx']);
    writetable(format_results(results), output_filename, 'WriteRowNames', true);
    disp(['Results written to ', output_filename]);

    output_filename_nozero = fullfile(save_path, ['PCAFM_results_', label, '_', control_prefix, '_nozero.xlsx']);
    writetable(format_results(results, true), output_filename_nozero, 'WriteRowNames', true);
    disp(['No-zero results written to ', output_filename_nozero]);
end

function results_table = format_results(results, remove_zero_beta)
    if nargin < 2
        remove_zero_beta = false;
    end

    if remove_zero_beta
        coefficients_table = results.Gammahat_nozero;
        variance_table = results.avarhat_nozero;
    else
        coefficients_table = results.Gammahat;
        variance_table = results.avarhat;
    end

    coefficients = coefficients_table{:,:};
    variance = variance_table{:,:};
    factor_names = coefficients_table.Properties.RowNames;

    if remove_zero_beta
        zero_beta_row_idx = strcmp(factor_names, 'Zero-beta');
        coefficients(zero_beta_row_idx, :) = [];
        factor_names(zero_beta_row_idx) = [];
    end

    standard_error = sqrt(variance);
    t_stat = coefficients ./ standard_error;

    coefficients = round(coefficients, 4);
    standard_error = round(standard_error, 4);
    t_stat = round(t_stat, 4);

    column_names = coefficients_table.Properties.VariableNames;  

    results_table = table('RowNames', factor_names);
    for i = 1:numel(column_names)
        col_name = column_names{i};  
        results_table.([col_name '_Beta']) = coefficients(:, i);
        results_table.([col_name '_SE']) = standard_error(:, i);
        results_table.([col_name '_tStat']) = t_stat(:, i);
    end
end
