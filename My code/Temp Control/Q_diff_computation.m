theta = 2;
eta = 5;
gamma_val = 0.9;
episode = 5000;
L_A = 1;

psi_update = 0;

function difference = chi(eta, L_A)
    beta = 0.999*eta;
    gamma = 0.00175*eta + 0.06*L_A*eta;
    difference = beta + gamma;
end

function result = chi_i(i, eta, L_A)
    result = eta;
    for j = 1:i
        result = chi(result, L_A);
    end
end

sum_term = 0;
for i = 1:(episode-1)
    chi_result = chi_i(i, eta, L_A);
    sum_term = sum_term + theta * gamma_val^i * chi_result;
end

Psi_episode = theta * eta + sum_term;

% Display the result
disp(['Psi^(' num2str(episode) ') = ' num2str(Psi_episode)]);