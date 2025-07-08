%% Visualisation; 
disp(' ')
disp('Step 3: Visualisation')
disp(' ')

% Set the maximum number of steps for visualization
max_visualization_steps = maxStepsPerEpisode; % max_steps

x1 = zeros(1, max_visualization_steps + 1);
x2 = zeros(1, max_visualization_steps + 1);

% Time vector
t = zeros(1, max_visualization_steps + 1);
t(1) = 0; % Start time

% Initial temparature of each room
x1(1) = 4; % Initial Armature current
x2(1) = 0.045; % Initial Rotor speed

% Simulate based on the optimal policy
for step = 1:max_visualization_steps
    
    state =  [x1(step), x2(step)];
    stateLine = discretizeState(state, stateGrid, stateDim, n_x);

    % Get the action from the optimal policy (you can modify this part)
    Control = optimalPolicy(stateLine, :);

    % Update the temparature
    NextState = update_environment(state, Control, samplingTime);
    x1(step+1) = NextState(1);
    x2(step+1) = NextState(2);
    t(step + 1) = step*samplingTime;
    
    subplot(2,1,1)
    hold on;
    fill([t(1:step+1), fliplr(t(1:step+1))],[goal_vertices(1,1) * ones(size(t(1:step+1))),...
            fliplr(goal_vertices(2,1) * ones(size(t(1:step+1))))],'g','FaceAlpha',0.3,'EdgeColor','none');
    plot(t(1:step+1), x1(1:step+1), 'ro', 'MarkerSize', 2, 'MarkerFaceColor', 'r');
    ylim([stateMin(1), stateMax(1)]);
    ylabel('Armature Current');
    xlabel('Time');
    grid on;
    hold off;
    
    subplot(2,1,2)
    hold on;
    fill([t(1:step+1), fliplr(t(1:step+1))],[goal_vertices(1,2) * ones(size(t(1:step+1))),...
            fliplr(goal_vertices(2,2) * ones(size(t(1:step+1))))],'g','FaceAlpha',0.3,'EdgeColor','none');
    plot(t(1:step+1), x2(1:step+1), 'ko', 'MarkerSize', 2, 'MarkerFaceColor', 'r');
    ylim([stateMin(2), stateMax(2)]);
    ylabel('Rotor speed');
    xlabel('Time');
    grid on;
    hold off

    % Check if the temparature is inside the desired domain
    is_inside_goal = (NextState(1) >= goal_vertices(1,1)) && (NextState(1) <= goal_vertices(2,1)) && ...
                     (NextState(2) >= goal_vertices(1,2)) && (NextState(2) <= goal_vertices(2,2));
    if is_inside_goal
        break; % High reward for reaching the goal
    end
end
%%
% Helper function to discretize state based on the state grid

function stateLine = discretizeState(x, stateGrid, stateDim, n_x)

stateIndex = zeros(stateDim, 1);

% Calculate the index for each dimension of the state
for i = 1:stateDim
    [~, idx] = min(abs(x(i) - stateGrid{i}));
    stateIndex(i) = idx;
end

stateLine = (stateIndex(1)-1)*(n_x) + stateIndex(2);

end

%%
function x_next = update_environment(x, u, Ts)
    % Define state space specifications
    x_min = [0; 0];
    x_max = [5; 0.5];
    
    % Compute intermediate variables
    Ra = 1;
    La = 0.01;
    J = 0.01;
    b = 1;
    kb = 0.01;

    % Update state using Euler integration
    x_next = zeros(2, 1);
    x_next(1) = max(min(x(1)*(1-Ra*Ts/La) - x(2)*kb*Ts/La + u*Ts/La, x_max(1)), x_min(1));
    x_next(2) = max(min(x(1)*kb*Ts/J + x(2)*(1-b*Ts/J), x_max(2)), x_min(2));
end
