coarse_idx = [4, 5, 6];
[fine_df, coarse_df] = compare_coarse_fine_qvalues(coarse_idx);

disp('Fine Q-values:');
disp(head(fine_df));

% disp('Coarse Q-values:');
% disp(coarse_df);

function [fine_table, coarse_table] = compare_coarse_fine_qvalues(coarse_idx, coarse_grid_size, fine_grid_size, action_bins)

    if nargin < 2
        coarse_grid_size = 25;
    end
    if nargin < 3
        fine_grid_size = 50;
    end
    if nargin < 4
        action_bins = 2;
    end

    % Load Q-tables
    qtable_fine = readtable('C:\Users\Ahan_FOCASLab\OneDrive - Indian Institute of Science\PhD Projects and Other\Ahan PhD work\Q learning using Incremental Stability\Q-learn-with-Inc.-Stability\Q-learn-ISS\My code\Temp Control\Qtable_25_structured.xlsx');
    qtable_coarse = readtable('C:\Users\Ahan_FOCASLab\OneDrive - Indian Institute of Science\PhD Projects and Other\Ahan PhD work\Q learning using Incremental Stability\Q-learn-with-Inc.-Stability\Q-learn-ISS\My code\Temp Control\Qtable_10_structured.xlsx');

    i = coarse_idx(1);
    j = coarse_idx(2);
    k = coarse_idx(3);

    fine_per_coarse = fine_grid_size / coarse_grid_size;

    % Generate list of fine cells within the coarse cell
    fine_cells = [];
    for dx = 0:fine_per_coarse-1
        for dy = 0:fine_per_coarse-1
            for dz = 0:fine_per_coarse-1
                fine_i = i * fine_per_coarse + dx;
                fine_j = j * fine_per_coarse + dy;
                fine_k = k * fine_per_coarse + dz;
                fine_cells = [fine_cells; fine_i, fine_j, fine_k];
            end
        end
    end

    % Initialize result for fine-level Q-values
    fine_result = [];

    % Extract fine Q-values for all fine cells and actions
    for idx = 1:size(fine_cells, 1)
        fx = fine_cells(idx, 1);
        fy = fine_cells(idx, 2);
        fz = fine_cells(idx, 3);

        for ax = 0:action_bins-1
            for ay = 0:action_bins-1
                for az = 0:action_bins-1
                    match = qtable_fine(...
                        qtable_fine.state_idx_x == fx & ...
                        qtable_fine.state_idx_y == fy & ...
                        qtable_fine.state_idx_z == fz & ...
                        qtable_fine.action_idx_x == ax & ...
                        qtable_fine.action_idx_y == ay & ...
                        qtable_fine.action_idx_z == az, :);

                    if ~isempty(match)
                        for r = 1:height(match)
                            fine_result = [fine_result; ...
                                table(fx, fy, fz, ax, ay, az, match.Q_value(r), ...
                                'VariableNames', {'fine_state_x', 'fine_state_y', 'fine_state_z', ...
                                                  'action_x', 'action_y', 'action_z', 'fine_Q'})];
                        end
                    end
                end
            end
        end
    end

    % Extract coarse Q-values for this coarse cell
    coarse_rows = qtable_coarse(...
        qtable_coarse.state_idx_x == i & ...
        qtable_coarse.state_idx_y == j & ...
        qtable_coarse.state_idx_z == k, :);

    % Build a map for coarse Q-values by action
    coarse_q_map = containers.Map();
    for r = 1:height(coarse_rows)
        key = sprintf('%d_%d_%d', coarse_rows.action_idx_x(r), coarse_rows.action_idx_y(r), coarse_rows.action_idx_z(r));
        coarse_q_map(key) = coarse_rows.("Q_value")(r);
    end

    % Add coarse Q-values to fine table
    coarse_Q_col = zeros(height(fine_result), 1);
    for idx = 1:height(fine_result)
        key = sprintf('%d_%d_%d', fine_result.action_x(idx), fine_result.action_y(idx), fine_result.action_z(idx));
        if isKey(coarse_q_map, key)
            coarse_Q_col(idx) = coarse_q_map(key);
        else
            coarse_Q_col(idx) = NaN;
        end
    end

    fine_result.coarse_Q = coarse_Q_col;

    % Also return just the coarse values per action
    coarse_result = [];
    keys = coarse_q_map.keys;
    for k = 1:length(keys)
        key_parts = sscanf(keys{k}, '%d_%d_%d');
        coarse_result = [coarse_result; ...
            table(i, j, k, key_parts(1), key_parts(2), key_parts(3), coarse_q_map(keys{k}), ...
            'VariableNames', {'coarse_state_x', 'coarse_state_y', 'coarse_state_z', ...
                              'action_x', 'action_y', 'action_z', 'coarse_Q'})];
    end

    fine_table = fine_result;
    coarse_table = coarse_result;

end
