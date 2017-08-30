function [W, H, P, cost] = cmfwisa_regularized(V, num_basis_elems, W_ref, H_ref, config)
% [W, H, P, cost] = cmfwisa_regularized(V, num_basis_elems, W_ref, H_ref, config) 
% Decompose a (complex-valued) matrix V into WH.*P using Complex
% NMF with intra-source additivity (CMF-WISA) [1] by minimizing the
% Euclidean distance between V and WH.*P. W is a basis matrix, H is the
% encoding matrix that encodes the input V in terms of the basis W, and P
% is the phase matrix. This function can output multiple
% basis/encoding/phase matrices for multiple sources, each of which can 
% be fixed to a given matrix or have a given sparsity level. With 1 source,
% CMF-WISA essentially becomes NMF.
%
% Inputs:
%   V: [matrix]
%       m-by-n matrix containing data, possibly complex-valued, to be decomposed.
%   num_basis_elems: [positive scalar] or [cell array]
%       [positive scalar]: number of basis elements (columns of W/rows of H)
%       for 1 source.
%       [cell array]: K-length array of positive scalars {num_basis_elems_1,
%       ...,num_basis_elems_K} specifying the number of basis elements for
%       K sources.
%   W_ref: [non-negative matrix] or [cell array]
%       [non-negative matrix]: reference basis matrix to use for the basis
%       regularization term.
%       [cell array]: K-length array of reference basis matrices to use for
%       the basis regularization term for K sources.
%   H_ref: [non-negative matrix] or [cell array]
%       [non-negative matrix]: reference encoding matrix to use for the
%       encoding matrix regularization term.
%       [cell array]: K-length array of reference encoding matrices to use
%       for the encoding matrix regularization term for K sources.
%   config: [structure] (optional)
%       structure containing configuration parameters.
%       config.W_init: [non-negative matrix] or [cell array] (default:
%           random matrix or K-length cell array of random matrices)
%           [non-negative matrix]: initialize 1 basis matrix for 1 source
%           with a m-by-num_basis_elems tensor.
%           [cell array]: initialize K basis matrices for K sources with a 
%           K-length array containing {m-by-num_basis_elems_1, ...,
%           m-by-num_basis_elems_K} non-negative matrices. 
%       config.H_init: [non-negative matrix] or [cell array] (default:
%           random matrix or K-length cell array of random matrices)
%           [non-negative matrix]: initialize 1 encoding matrix for 1
%           source with a num_basis_elems-by-n non-negative matrix.
%           [cell array]: initialize K encoding matrices for K sources with
%           a K-length array containing {num_basis_elems_1-by-n, ...,
%           num_basis_elems_K-by-n} non-negative matrices.
%       config.P_init: [matrix] or [cell array] (default: exp(1j * arg(V))
%           or K-length cell array of exp(1j * arg(V)) replicated K times)
%           [matrix]: initialize 1 phase matrix for 1 source with a m-by-n
%           (complex-valued) matrix.
%           [cell array]: initialize K phase matrices for K sources with a
%           K-length array containing {m-by-n,...,m-by-n} (complex-valued)
%           matrices.
%       config.W_sparsity: [non-negative scalar] or [cell array] (default: 0)
%           [non-negative scalar]: sparsity level for all basis matrices.
%           [cell array]: K-length array of non-negative scalars indicating
%           the sparsity level of the K basis matrices.
%       config.H_sparsity: [non-negative scalar] or [cell array] (default: 0)
%           [non-negative scalar]: sparsity level for all K encoding matrices.
%           [cell array]: K-length array of non-negative scalars indicating
%           the sparsity level of the K encoding matrices.
%       config.W_fixed: [boolean] or [cell array] (default: false)
%           [boolean]: indicate if all basis matrices are fixed during the
%           update equations.
%           [cell array] K-length array of booleans indicating if the
%           corresponding basis matrices are fixed during the update equations.
%       config.H_fixed: [boolean] or [cell array] (default: false)
%           [boolean]: indicate if all encoding matrices are fixed during
%           the update equations.
%           [cell array] K-length array of booleans indicating if the
%           corresponding encoding matrices are fixed during the update equations.
%       config.P_fixed: [boolean] or [cell array] (default: false)
%           [boolean]: indicate if all phase matrices are fixed during
%           the update equations.
%           [cell array] K-length array of booleans indicating if the
%           corresponding phase matrices are fixed during the update equations.
%       config.W_regularization: [non-negative matrix] or [cell array]
%           (default: m-length vector of zeros)
%           [non-negative matrix]: matrix of m rows indicating how similar
%           each row in W should be to W_ref. Higher values enforce greater
%           similarity. Multiple columns allow for multiple basis
%           regularizations applied simultaneously.
%           [cell array]: K-length array of non-negative matrices
%           indicating the amount of basis regularization for K sources.
%       config.H_regularization: [non-negative scalar] or [cell array]
%           (default: 0)
%           [non-negative scalar]: amount of encoding matrix regularization
%           to use when learning the encoding matrix.
%           [cell array]: K-length array of non-negative scalars indicating
%           the amount of encoding matrix regularization for K sources.
%       config.maxiter: [positive scalar] (default: 100)
%           maximum number of update iterations.
%       config.tolerance: [positive scalar] (default: 1e-3)
%           maximum change in the cost function between iterations before
%           the algorithm is considered to have converged.
%
% Outputs:
%   W: [non-negative matrix] or [cell array]
%       [non-negative matrix]: m-by-num_basis_elems non-negative basis matrix.    
%       [cell array]: K-length array containing {m-by-num_basis_elems_1, ...,
%       m-by-num_basis_elems_K} non-negative basis matrices for K sources.
%   H: [non-negative matrix] or [cell array]
%       [non-negative matrix]: num_basis_elems-by-n non-negative encoding matrix.
%       [cell array]: K-length array containing {num_basis_elems_1-by-n, ...,
%       num_basis_elems_K-by-n} non-negative encoding matrices.
%   P: [matrix] or [cell array]
%       [matrix]: m-by-n (complex-valued) phase matrix.
%       [cell array]: K-length array containing {m-by-n,...,m-by-n}
%       (complex-valued) phase matrices.
%   cost: [vector]
%       value of the cost function after each iteration.
%
% References:
%   [1] B. King, "New Methods of Complex Matrix Factorization for
%       Single-Channel Source Separation and Analysis," Ph.D. thesis,
%       University of Washington, Seattle, WA, 2012.
%
% NMF Toolbox
% Colin Vaz - cvaz@usc.edu
% Signal Analysis and Interpretation Lab (SAIL) - http://sail.usc.edu
% University of Southern California
% 2015

if nargin < 5
    config = struct;
end

% Validate parameters
[m, n] = size(V);
if ~iscell(num_basis_elems)
    num_basis_elems = {num_basis_elems};
end
num_sources = length(num_basis_elems);
if ~iscell(W_ref)
    W_ref = {W_ref};
end
if ~iscell(H_ref)
    H_ref = {H_ref};
end
[config, is_W_cell, is_H_cell, is_P_cell] = ValidateParameters(config, V, num_basis_elems, W_ref, H_ref);

% Set variables
W = config.W_init;
H = config.H_init;
P = config.P_init;

beta = cell(num_sources, 1);
V_hat_per_source = zeros(m, n, num_sources);
for i = 1 : num_sources
    V_hat_per_source(:, :, i) = (W{i} * H{i}) .* P{i};
end

V_hat = sum(V_hat_per_source, 3);
V_bar_per_source = zeros(m, n, num_sources);

% Set up basis matrix regularization
num_basis_regularizations = zeros(num_sources, 1);
replication_idx = cell(num_sources, 1);
for i = 1 : num_sources
    num_basis_regularizations(i) = size(config.W_regularization{i}, 2);
    W_ref{i} = repmat(W_ref{i}, 1, num_basis_regularizations(i));
    H_ref{i} = repmat(H_ref{i} / num_basis_regularizations(i), num_basis_regularizations(i), 1);
    W{i} = repmat(W{i}, 1, num_basis_regularizations(i));
    H{i} = repmat(H{i} / num_basis_regularizations(i), num_basis_regularizations(i), 1);
    replication_idx{i} = zeros(num_basis_elems{i}, num_basis_regularizations(i));
    for rep = 1 : num_basis_regularizations(i)
        replication_idx{i}(:, rep) = [(rep-1)*num_basis_elems{i} + 1 : rep*num_basis_elems{i}]';
    end
end

% Set up encoding matrix regularization
ref_mean = cell(num_sources, 1);
ref_var = cell(num_sources, 1);
ref_mean_pos = cell(num_sources, 1);
ref_mean_neg = cell(num_sources, 1);
curr_mean = cell(num_sources, 1);
curr_var = cell(num_sources, 1);
curr_mean_pos = cell(num_sources, 1);
curr_mean_neg = cell(num_sources, 1);
logH_pos = cell(num_sources, 1);
logH_neg = cell(num_sources, 1);
for i = 1 : num_sources
    ref_mean{i} = mean(log(H_ref{i}), 2);
    ref_var{i} = var(log(H_ref{i}), [], 2);
    ref_mean_pos{i} = 0.5 * diag(abs(ref_mean{i}) + ref_mean{i});
    ref_mean_neg{i} = 0.5 * diag(abs(ref_mean{i}) - ref_mean{i});
    curr_mean{i} = mean(log(H{i}), 2);
    curr_var{i} = var(log(H{i}), [], 2);
    curr_mean_pos{i} = 0.5 * diag(abs(curr_mean{i}) + curr_mean{i});
    curr_mean_neg{i} = 0.5 * diag(abs(curr_mean{i}) - curr_mean{i});
    logH_pos{i} = 0.5 * (abs(log(H{i})) + log(H{i}));
    logH_neg{i} = 0.5 * (abs(log(H{i})) - log(H{i}));
end

cost = zeros(config.maxiter, 1);

% Begin iterations to minimize cost function
for iter = 1 : config.maxiter
    W_all = cell2mat(W);
    H_all = cell2mat(H);

    % Update auxiliary variables
    for i = 1 : num_sources
        beta{i} = (W{i} * H{i}) ./ (W_all * H_all);
        V_bar_per_source(:, :, i) = V_hat_per_source(:, :, i) + beta{i} .* (V - V_hat);
    end
    
    % Update phase matrices
    for i = 1 : num_sources
        if ~config.P_fixed{i}
            P{i} = exp(1j * angle(V_bar_per_source(:, :, i))); %V_bar_per_source(:, :, i) ./ abs(V_bar_per_source(:, :, i));
        end
    end
    
    % Update basis matrices
    for i = 1 : num_sources
        if ~config.W_fixed{i}
            for rep = 1 : num_basis_regularizations(i)
                basis_reg_neg = diag(config.W_regularization{i}(:, rep))' * diag(config.W_regularization{i}(:, rep)) * W_ref{i}(:, replication_idx{i}(:, rep));
                basis_reg_pos = diag(config.W_regularization{i}(:, rep))' * diag(config.W_regularization{i}(:, rep)) * W{i}(:, replication_idx{i}(:, rep));
                W{i}(:, replication_idx{i}(:, rep)) = W{i}(:, replication_idx{i}(:, rep)) .* (((abs(V_bar_per_source(:, :, i)) ./ max(beta{i}, eps)) * H{i}(replication_idx{i}(:, rep), :)' + basis_reg_neg) ./ ...
                                                                                              max(W_all * H_all * H{i}(replication_idx{i}(:, rep), :)' + basis_reg_pos + config.W_sparsity{i}, eps));
            end
            W{i} = W{i} * diag(1 ./ sqrt(sum(W{i}.^2, 1)));
        end
    end
    
    % Update encoding matrices
    for i = 1 : num_sources
        if ~config.H_fixed{i}
            encoding_reg_neg = (1 ./ H{i}) .* (((1/n) * diag(1 ./ curr_var{i}) * (ref_mean_pos{i} + curr_mean_neg{i}) * ones(num_basis_regularizations(i) * num_basis_elems{i}, n) + (1/(n-1)) * diag((ref_var{i} + (curr_mean{i} - ref_mean{i}).^2) ./ curr_var{i}.^2) * (logH_pos{i} + curr_mean_neg{i} * ones(num_basis_regularizations(i) * num_basis_elems{i}, n)) + (1/(n-1)) * diag(1 ./ curr_var{i}) * (logH_neg{i} + curr_mean_pos{i} * ones(num_basis_regularizations(i) * num_basis_elems{i}, n))));
            encoding_reg_pos = (1 ./ H{i}) .* (((1/n) * diag(1 ./ curr_var{i}) * (ref_mean_neg{i} + curr_mean_pos{i}) * ones(num_basis_regularizations(i) * num_basis_elems{i}, n) + (1/(n-1)) * diag((ref_var{i} + (curr_mean{i} - ref_mean{i}).^2) ./ curr_var{i}.^2) * (logH_neg{i} + curr_mean_pos{i} * ones(num_basis_regularizations(i) * num_basis_elems{i}, n)) + (1/(n-1)) * diag(1 ./ curr_var{i}) * (logH_pos{i} + curr_mean_neg{i} * ones(num_basis_regularizations(i) * num_basis_elems{i}, n))));
            H{i} = H{i} .* ((W{i}' * (abs(V_bar_per_source(:, :, i)) ./ max(beta{i}, eps)) + config.H_regularization{i} * encoding_reg_neg) ./ ...
                         max(W{i}' * W_all * H_all + config.H_regularization{i} * encoding_reg_pos + config.H_sparsity{i}, eps)); %max(H .* ((W.^2)' * (ones(m, n) ./ beta)) + config.H_sparsity, eps));
            H{i} = max(H{i}, eps);
        end
    end
    
    for i = 1 : num_sources
        curr_mean{i} = mean(log(H{i}), 2);
        curr_var{i} = var(log(H{i}), [], 2);
        curr_mean_pos{i} = 0.5 * diag(abs(curr_mean{i}) + curr_mean{i});
        curr_mean_neg{i} = 0.5 * diag(abs(curr_mean{i}) - curr_mean{i});
        logH_pos{i} = 0.5 * (abs(log(H{i})) + log(H{i}));
        logH_neg{i} = 0.5 * (abs(log(H{i})) - log(H{i}));
        V_hat_per_source(:, :, i) = (W{i} * H{i}) .* P{i};
    end
    
    V_hat = sum(V_hat_per_source, 3);

    % Calculate cost for this iteration
    cost(iter) = 0.5 * sum(sum(abs(V - V_hat).^2));
    for i = 1 : num_sources
        for rep = 1 : num_basis_regularizations(i)
            cost(iter) = cost(iter) + 0.5 * norm(diag(config.W_regularization{i}(:, rep)) * (W_ref{i}(:, replication_idx{i}(:, rep)) - W{i}(:, replication_idx{i}(:, rep))), 'fro')^2;
        end
        cost(iter) = cost(iter) + 0.5 * (sum(ref_var{i} ./ curr_var{i}) + ((curr_mean{i} - ref_mean{i})' / diag(curr_var{i})) * (curr_mean{i} - ref_mean{i}) - num_basis_regularizations(i) * num_basis_elems{i} + sum(log(curr_var{i})) - sum(log(ref_var{i})));
        cost(iter) = cost(iter) + config.W_sparsity{i} * sum(sum(W{i})) + config.H_sparsity{i} * sum(sum(H{i}));
    end
    
    % Stop iterations if change in cost function less than the tolerance
    if iter > 1 && cost(iter) < cost(iter-1) && cost(iter-1) - cost(iter) < config.tolerance
        cost = cost(1 : iter);  % trim vector
        break;
    end
end

% Prepare the output
if ~is_W_cell
    W = W{1};
end

if ~is_H_cell
    H = H{1};
end

if ~is_P_cell
    P = P{1};
end

end  % function

function [config_out, is_W_cell, is_H_cell, is_P_cell] = ValidateParameters(config_in, V, num_basis_elems, W_ref, H_ref)

config_out = config_in;
[m, n] = size(V);
num_sources = length(num_basis_elems);

% Check validity of given reference basis matrices
if length(W_ref) ~= num_sources
    error(['Requested ', num2str(num_sources), ' sources. Given ', num2str(length(W_ref)), ' reference basis matrices.']);
end
for i = 1 : num_sources
    if size(W_ref{i}, 1) ~= m
        error(['Dimension of reference basis ', num2str(i), ' (dim = ', size(W_ref{i}, 1), ') does not match dimension of input (dim = ', num2str(m), ').']);
    elseif size(W_ref{i}, 2) ~= num_basis_elems{i}
        error(['Number of vectors in reference basis ', num2str(i), ' (nvec = ', size(W_ref{i}, 2), ') does not match requested number of basis elements (', num2str(num_basis_elems{i}), ').']);
    end
end

% Check validity of given reference encoding matrices
if length(H_ref) ~= num_sources
    error(['Requested ', num2str(num_sources), ' sources. Given ', num2str(length(H_ref)), ' reference encoding matrices.']);
end
for i = 1 : num_sources
    if size(H_ref{i}, 1) ~= num_basis_elems{i}
        error(['Number of rows in reference encoding matrix ', num2str(i), ' (nvec = ', size(H_ref{i}, 1), ') does not match requested number of basis elements (', num2str(num_basis_elems{i}), ').']);
    end
end

% Initialize basis matrices
if ~isfield(config_out, 'W_init') || isempty(config_out.W_init)  % not given any inital basis matrices. Fill these in.
    if num_sources == 1
        is_W_cell = false;
    else
        is_W_cell = true;
    end
    config_out.W_init = cell(1, num_sources);
    for i = 1 : num_sources
        config_out.W_init{i} = max(rand(m, num_basis_elems{i}), eps);
        config_out.W_init{i} = config_out.W_init{i} * diag(1 ./ sqrt(sum(config_out.W_init{i}.^2, 1)));
    end
elseif iscell(config_out.W_init) && length(config_out.W_init) ~= num_sources  % given an incorrect number of initial basis matrices
    error(['Requested ', num2str(num_sources), ' sources. Given ', num2str(length(config_out.W_init)), ' initial basis matrices.']);
elseif ~iscell(config_out.W_init)  % given a matrix
    is_W_cell = false;
    config_out.W_init = {config_out.W_init};
else  % organize basis matrices as {W_1 W_2 ... W_num_bases}
    is_W_cell = true;
    config_out.W_init = config_out.W_init(:)';
end

% Initialize encoding matrices
if ~isfield(config_out, 'H_init') || isempty(config_out.H_init)  % not given any inital encoding matrices. Fill these in.
    if num_sources == 1
        is_H_cell = false;
    else
        is_H_cell = true;
    end
    config_out.H_init = cell(num_sources, 1);
    for i = 1 : num_sources
        config_out.H_init{i} = max(rand(num_basis_elems{i}, n), eps);
    end
elseif iscell(config_out.H_init) && length(config_out.H_init) ~= num_sources  % given an incorrect number of initial encoding matrices
    error(['Requested ', num2str(num_sources), ' sources. Given ', num2str(length(config_out.H_init)), ' initial encoding matrices.']);
elseif ~iscell(config_out.H_init)  % given a matrix
    is_H_cell = false;
    config_out.H_init = {config_out.H_init};
else  % organize encoding matrices as {H_1; H_2; ...; H_num_bases}
    is_H_cell = true;
    config_out.H_init = config_out.H_init(:);
end

% Initialize phase matrices
if ~isfield(config_out, 'P_init') || isempty(config_out.P_init)  % not given any inital phase matrices. Fill these in.
    if num_sources == 1
        is_P_cell = false;
    else
        is_P_cell = true;
    end
    config_out.P_init = cell(num_sources, 1);
    for i = 1 : num_sources
        config_out.P_init{i} = exp(1j * angle(V)); %V ./ abs(V);
    end
elseif iscell(config_out.P_init) && length(config_out.P_init) ~= num_sources  % given an incorrect number of initial phase matrices
    error(['Requested ', num2str(num_sources), ' encoding matrices. Given ', num2str(length(config_out.P_init)), ' initial phase matrices.']);
elseif ~iscell(config_out.P_init)  % given a matrix
    is_P_cell = false;
    config_out.P_init = {config_out.P_init};
else  % organize phase matrices as {P_1; P_2; ...; P_num_sources}
    is_P_cell = true;
    config_out.P_init = config_out.P_init(:);
end

% Sparsity levels for basis matrices
if ~isfield(config_out, 'W_sparsity') || isempty(config_out.W_sparsity)  % not given a sparsity level. Fill this in.
    config_out.W_sparsity = cell(num_sources, 1);
    for i = 1 : num_sources
        config_out.W_sparsity{i} = 0;
    end
elseif iscell(config_out.W_sparsity) && length(config_out.W_sparsity) > 1 && length(config_out.W_sparsity) ~= num_sources  % given an incorrect number of sparsity levels
    error(['Requested ', num2str(num_sources), ' sources. Given ', num2str(length(config_out.W_sparsity)), ' sparsity levels.']);
elseif ~iscell(config_out.W_sparsity)  || length(config_out.W_sparsity) == 1  % extend one sparsity level to all basis matrices
    if iscell(config_out.W_sparsity)
        temp = max(config_out.W_sparsity{1}, 0);
    else
        temp = max(config_out.W_sparsity, 0);
    end
    config_out.W_sparsity = cell(num_sources, 1);
    for i = 1 : num_sources
        config_out.W_sparsity{i} = temp;
    end
    clear temp;
else  % make sure all given sparsity levels are non-negative
    for i = 1 : num_sources
        config_out.W_sparsity{i} = max(config_out.W_sparsity{i}, 0);
    end
end

% Sparsity levels for encoding matrices
if ~isfield(config_out, 'H_sparsity') || isempty(config_out.H_sparsity)  % not given a sparsity level. Fill this in.
    config_out.H_sparsity = cell(num_sources, 1);
    for i = 1 : num_sources
        config_out.H_sparsity{i} = 0;
    end
elseif iscell(config_out.H_sparsity) && length(config_out.H_sparsity) > 1 && length(config_out.H_sparsity) ~= num_sources  % given an incorrect number of sparsity levels
    error(['Requested ', num2str(num_sources), ' sources. Given ', num2str(length(config_out.H_sparsity)), ' sparsity levels.']);
elseif ~iscell(config_out.H_sparsity)  || length(config_out.H_sparsity) == 1  % extend one sparsity level to all encoding matrices
    if iscell(config_out.H_sparsity)
        temp = max(config_out.H_sparsity{1}, 0);
    else
        temp = max(config_out.H_sparsity, 0);
    end
    config_out.H_sparsity = cell(num_sources, 1);
    for i = 1 : num_sources
        config_out.H_sparsity{i} = temp;
    end
    clear temp;
else  % make sure all given sparsity levels are non-negative
    for i = 1 : num_sources
        config_out.H_sparsity{i} = max(config_out.H_sparsity{i}, 0);
    end
end

% Update switches for basis matrices
if ~isfield(config_out, 'W_fixed') || isempty(config_out.W_fixed)  % not given an update switch. Fill this in.
    config_out.W_fixed = cell(num_sources, 1);
    for i = 1 : num_sources
        config_out.W_fixed{i} = false;
    end
elseif iscell(config_out.W_fixed) && length(config_out.W_fixed) > 1 && length(config_out.W_fixed) ~= num_sources  % given an incorrect number of update switches
    error(['Requested ', num2str(num_sources), ' sources. Given ', num2str(length(config_out.W_fixed)), ' update switches.']);
elseif ~iscell(config_out.W_fixed)  || length(config_out.W_fixed) == 1  % extend one update switch level to all basis matrices
    if iscell(config_out.W_fixed)
        temp = config_out.W_fixed{1};
    else
        temp = config_out.W_fixed;
    end
    config_out.W_fixed = cell(num_sources, 1);
    for i = 1 : num_sources
        config_out.W_fixed{i} = temp;
    end
    clear temp;
end

% Update switches for encoding matrices
if ~isfield(config_out, 'H_fixed') || isempty(config_out.H_fixed)  % not given an update switch. Fill this in.
    config_out.H_fixed = cell(num_sources, 1);
    for i = 1 : num_sources
        config_out.H_fixed{i} = false;
    end
elseif iscell(config_out.H_fixed) && length(config_out.H_fixed) > 1 && length(config_out.H_fixed) ~= num_sources  % given an incorrect number of update switches
    error(['Requested ', num2str(num_sources), ' sources. Given ', num2str(length(config_out.H_fixed)), ' update switches.']);
elseif ~iscell(config_out.H_fixed)  || length(config_out.H_fixed) == 1  % extend one update switch level to all encoding matrices
    if iscell(config_out.H_fixed)
        temp = config_out.H_fixed{1};
    else
        temp = config_out.H_fixed;
    end
    config_out.H_fixed = cell(num_sources, 1);
    for i = 1 : num_sources
        config_out.H_fixed{i} = temp;
    end
    clear temp;
end

% Update switches for phase matrices
if ~isfield(config_out, 'P_fixed') || isempty(config_out.P_fixed)  % not given an update switch. Fill this in.
    config_out.P_fixed = cell(num_sources, 1);
    for i = 1 : num_sources
        config_out.P_fixed{i} = false;
    end
elseif iscell(config_out.P_fixed) && length(config_out.P_fixed) > 1 && length(config_out.P_fixed) ~= num_sources  % given an incorrect number of update switches
    error(['Requested ', num2str(num_sources), ' basis matrices. Given ', num2str(length(config_out.P_fixed)), ' update switches.']);
elseif ~iscell(config_out.P_fixed)  || length(config_out.P_fixed) == 1  % extend one update switch level to all phase matrices
    if iscell(config_out.P_fixed)
        temp = config_out.P_fixed{1};
    else
        temp = config_out.P_fixed;
    end
    config_out.P_fixed = cell(num_sources, 1);
    for i = 1 : num_sources
        config_out.P_fixed{i} = temp;
    end
    clear temp;
end

% Regularization for basis matrices
% TODO: check that dimension of W_regularization == m
if ~isfield(config_out, 'W_regularization') || isempty(config_out.W_regularization)  % not given any regularization weights. Fill these in.
    config_out.W_regularization = cell(num_sources, 1);
    for i = 1 : num_sources
        config_out.W_regularization{i} = zeros(m, 1);
    end
elseif iscell(config_out.W_regularization) && length(config_out.W_regularization) ~= num_sources  % given an incorrect number of regularization weights
    error(['Requested ', num2str(num_sources), ' sources. Given ', num2str(length(config_out.W_regularization)), ' basis regularization weights.']);
elseif ~iscell(config_out.W_regularization)  || length(config_out.W_regularization) == 1  % extend one regularization weight to all basis matrices
    if iscell(config_out.W_regularization)
        temp = max(config_out.W_regularization{1}, zeros(m, size(config_out.W_regularization{1}, 2)));
    else
        temp = max(config_out.W_regularization, zeros(m, size(config_out.W_regularization, 2)));
    end
    config_out.W_regularization = cell(num_sources, 1);
    for i = 1 : num_sources
        config_out.W_regularization{i} = temp;
    end
    clear temp;
else  % make sure all given regularization weights are non-negative
    for i = 1 : num_sources
        config_out.W_regularization{i} = max(config_out.W_regularization{i}, zeros(m, size(config_out.W_regularization{i}, 2)));
    end
end

% Regularization for encoding matrices
if ~isfield(config_out, 'H_regularization') || isempty(config_out.H_regularization)  % not given any regularization weights. Fill these in.
    config_out.H_regularization = cell(num_sources, 1);
    for i = 1 : num_sources
        config_out.H_regularization{i} = 0;
    end
elseif iscell(config_out.H_regularization) && length(config_out.H_regularization) ~= num_sources  % given an incorrect number of regularization weights
    error(['Requested ', num2str(num_sources), ' sources. Given ', num2str(length(config_out.H_regularization)), ' basis regularization weights.']);
elseif ~iscell(config_out.H_regularization)  || length(config_out.H_regularization) == 1  % extend one regularization weight to all encoding matrices
    if iscell(config_out.H_regularization)
        temp = max(config_out.H_regularization{1}, 0);
    else
        temp = max(config_out.H_regularization, 0);
    end
    config_out.H_regularization = cell(num_sources, 1);
    for i = 1 : num_sources
        config_out.H_regularization{i} = temp;
    end
    clear temp;
else  % make sure all given regularization weights are non-negative
    for i = 1 : num_sources
        config_out.H_regularization{i} = max(config_out.H_regularization{i}, 0);
    end
end

% Maximum number of update iterations
if ~isfield(config_out, 'maxiter') || config_out.maxiter <= 0
    config_out.maxiter = 100;
end

% Maximum tolerance in cost function change per iteration
if ~isfield(config_out, 'tolerance') || config_out.tolerance <= 0
    config_out.tolerance = 1e-3;
end

end  % function
