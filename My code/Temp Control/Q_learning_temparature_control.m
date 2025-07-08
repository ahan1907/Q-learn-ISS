clc; 

% Define the vertices of the desired temparature domain
goal_vertices = [15 15 15; 20 20 20];
Treshold=0.0;

% Define state and action space dimensions
stateDim = 3;
actionDim = 3;

% Define state and action space boundaries
stateMin = [-10; -10; -10];
stateMax = [40; 40; 40];
actionMin = [0; 0; 0];
actionMax = [1; 1; 1];

% Discretize state and action spaces (adjust based on your needs)
n_x = 10;
n_modes = 2;
samplingTime = 0.2;

stateGrid = cell(stateDim, 1);
actionGrid = cell(actionDim, 1);
for i = 1:stateDim
  stateGrid{i} = linspace(stateMin(i), stateMax(i), n_x);
end
for i = 1:actionDim
  actionGrid{i} = linspace(actionMin(i), actionMax(i), n_modes);
end

% Define Q-table
Q_25 = zeros(n_x^stateDim, n_modes^actionDim);

% Set training parameters
maxEpisodes = 8000;
maxStepsPerEpisode = 6000;
learningRate = 0.9;
discountFactor = 0.99;
epsilon = 0.99; % Exploration rate
goal_reward = 0;

% Training loop
for i = 1:maxEpisodes
  % Reset environment (assuming initial state is 0)
  for j = 1: stateDim
      state(j) = stateMin(j) + (stateMax(j) - stateMin(j))*rand(1);
  end

  for t = 1:maxStepsPerEpisode
    % Discretize state
    stateLine = discretizeState(state, stateGrid, stateDim, n_x);
    
    % Choose action based on epsilon-greedy exploration
    if rand < epsilon
      action = randi(n_modes, 1);
    else
      [~, action] = max(Q_25(stateLine, :));
    end

    % Convert the linear index to row and column indices
    [row_idx, col_idx, dep_idx] = ind2sub([n_modes, n_modes, n_modes], action);
    
    % Convert action from index to continuous value
    continuousAction = [actionGrid{1}(row_idx) actionGrid{2}(col_idx), actionGrid{3}(dep_idx)];
    
    % Take action and observe reward (replace with your reward function)
    nextX = update_environment(state, continuousAction, samplingTime); % Ts = 0.1 assumed

    % Example reward
    % Check if the temparature is inside the desired domain
    is_inside_goal = (nextX(1) >= goal_vertices(1,1)) && (nextX(1) <= goal_vertices(2,1)) && ...
                     (nextX(2) >= goal_vertices(1,2)) && (nextX(2) <= goal_vertices(2,2)) && ...
                     (nextX(3) >= goal_vertices(1,3)) && (nextX(3) <= goal_vertices(2,3));

    if is_inside_goal
        reward = 0;
    else
        reward = -10; 
    end

    % Discretize next state
    nextXLine = discretizeState(nextX, stateGrid, stateDim, n_x);

    % Update Q-value using Q-learning update rule
    qValue = Q_25(stateLine, action);
    maxQnext = max(Q_25(nextXLine, :));
    newQValue = qValue + learningRate * (reward + discountFactor * maxQnext - qValue);
    Q_25(stateLine, action) = newQValue;

    % Update state for next step
    state = nextX;
    
    % Stop episode if terminal state reached (add your termination condition)
    if reward == 0
        % Print episode information (optional)
        fprintf('Episode %d: Steps %d: Reward %.2f\n', i, t, reward);
        break;
    end

  end

  % Update exploration rate (optional, can be decayed over time)
  epsilon = epsilon * 0.99;
  
 
  % Print episode information (optional)
  % fprintf('Episode %d: Reward %.2f\n', i, reward);
end


% Extract optimal policy from Q-table
optimalPolicy_25 = zeros(n_x^stateDim, actionDim);
for stateLine = 1:n_x^stateDim
    % Find the action with the highest Q-value for the current state
    [~, optimalAction] = max(Q_25(stateLine, :));
    
    % Convert the linear index of the action to row and column indices
    [row_idx, col_idx, dep_idx] = ind2sub([n_modes, n_modes, n_modes], optimalAction);
    
    % Convert action from index to continuous value
    continuousAction = [actionGrid{1}(row_idx), actionGrid{2}(col_idx), actionGrid{3}(dep_idx)];
    
    % Store the optimal action for the current state
    optimalPolicy_25(stateLine, :) = continuousAction;
end

qtable_data = [];

for stateLine = 1:n_x^stateDim
    % Convert linear state index to 3D indices
    [i, j, k] = ind2sub([n_x, n_x, n_x], stateLine);

    for a = 1:n_modes^actionDim
        [ai, aj, ak] = ind2sub([n_modes, n_modes, n_modes], a);
        q_value = Q_25(stateLine, a);

        % Append [state indices, action indices, Q-value]
        qtable_data = [qtable_data; i, j, k, ai, aj, ak, q_value];
    end
end

% Save structured Q-table
qtable_header = {'state_idx_x', 'state_idx_y', 'state_idx_z', ...
                 'action_idx_x', 'action_idx_y', 'action_idx_z', 'Q_value'};

% Write with header
writecell([qtable_header; num2cell(qtable_data)], 'Qtable_25_structured.xlsx');

%% Visualisation; 
disp(' ')
disp('Step 3: Visualisation')
disp(' ')

% Set the maximum number of steps for visualization
max_visualization_steps = maxStepsPerEpisode; % max_steps

x1 = zeros(1, max_visualization_steps + 1);
x2 = zeros(1, max_visualization_steps + 1);
x3 = zeros(1, max_visualization_steps + 1);

% Time vector
t = zeros(1, max_visualization_steps + 1);
t(1) = 0; % Start time

% Initial temparature of each room
x1(1) = 5; % Initial temparature of room 1
x2(1) = -6; % Initial temparature of room 2
x3(1) = 22; % Initial temparature of room 3

% Simulate the vehicle movement based on the optimal policy
for step = 1:max_visualization_steps
    
    state =  [x1(step), x2(step), x3(step)];
    stateLine = discretizeState(state, stateGrid, stateDim, n_x);

    % Get the action from the optimal policy (you can modify this part)
    Control = optimalPolicy_25(stateLine, :);

    % Update the temparature
    NextState= update_environment(state, Control, samplingTime);
    x1(step+1) = NextState(1);
    x2(step+1) = NextState(2);
    x3(step+1) = NextState(3);
    t(step + 1) = step*samplingTime;

    % Plot the trajectories as well as goal regions
    t = linspace(0,20, length(x1));
    
    subplot(3,1,1)
    hold on;
    fill([t(1:step+1), fliplr(t(1:step+1))],[goal_vertices(1,1) * ones(size(t(1:step+1))),...
            fliplr(goal_vertices(2,1) * ones(size(t(1:step+1))))],'g','FaceAlpha',0.3,'EdgeColor','none');
    plot(t(1:step+1), x1(1:step+1), 'ro', 'MarkerSize', 2, 'MarkerFaceColor', 'r');
    ylim([stateMin(1), stateMax(1)]);
    ylabel('Room Temp T1');
    xlabel('Time');
    grid on;
    hold off;
    
    subplot(3,1,2)
    hold on;
    fill([t(1:step+1), fliplr(t(1:step+1))],[goal_vertices(1,2) * ones(size(t(1:step+1))),...
            fliplr(goal_vertices(2,2) * ones(size(t(1:step+1))))],'g','FaceAlpha',0.3,'EdgeColor','none');
    plot(t(1:step+1), x2(1:step+1), 'ko', 'MarkerSize', 2, 'MarkerFaceColor', 'r');
    ylim([stateMin(2), stateMax(2)]);
    ylabel('Room Temp T2');
    xlabel('Time');
    grid on;
    hold off
    
    subplot(3,1,3)
    hold on;
    fill([t(1:step+1), fliplr(t(1:step+1))],[goal_vertices(1,3) * ones(size(t(1:step+1))),...
            fliplr(goal_vertices(2,3) * ones(size(t(1:step+1))))],'g','FaceAlpha',0.3,'EdgeColor','none');
    plot(t(1:step+1), x3(1:step+1), 'bo', 'MarkerSize', 2, 'MarkerFaceColor', 'r');
    ylim([stateMin(3), stateMax(3)]);
    ylabel('Room Temp T3');
    xlabel('Time');
    grid on;
    hold off;

    % Check if the temparature is inside the desired domain
    is_inside_goal = (NextState(1) >= goal_vertices(1,1)) && (NextState(1) <= goal_vertices(2,1)) && ...
                     (NextState(2) >= goal_vertices(1,2)) && (NextState(2) <= goal_vertices(2,2)) && ...
                     (NextState(3) >= goal_vertices(1,3)) && (NextState(3) <= goal_vertices(2,3));
    if is_inside_goal
        break; % High reward for reaching the goal
    end
end

% saveas(gcf, 'Temp_control_eps=1.png')
%%
% Helper function to discretize state based on the state grid

function stateLine = discretizeState(x, stateGrid, stateDim, n_x)

stateIndex = zeros(stateDim, 1);

% Calculate the index for each dimension of the state
for i = 1:stateDim
    [~, idx] = min(abs(x(i) - stateGrid{i}));
    stateIndex(i) = idx;
end

% stateLine = (stateIndex(1)-1)*(n_x) + stateIndex(2);
stateLine = sub2ind([n_x n_x n_x], stateIndex(1), stateIndex(2), stateIndex(3));
end

%%
function x_next = update_environment(x, u, Ts)
    % Define state space specifications
    x_min = [-10; -10; -10];
    x_max = [40; 40; 40];
    
    % Compute intermediate variables
    a = 1/20;
    b = 1/200;
    g = 1/100;
    t_e = 0;
    t_h = 50;

    % Update state using Euler integration
    x_next = zeros(3, 1);
    x_next(1) = max(min(x(1)*(1-2*a*Ts-b*Ts) + x(2)*a*Ts + x(3)*a*Ts + u(1)*g*Ts*(t_h-x(1)) + b*Ts*t_e, x_max(1)), x_min(1));
    x_next(2) = max(min(x(1)*a*Ts + x(2)*(1-2*a*Ts-b*Ts) + x(3)*a*Ts + u(2)*g*Ts*(t_h-x(2)) + b*Ts*t_e, x_max(2)), x_min(2));
    x_next(3) = max(min(x(1)*a*Ts + x(2)*a*Ts + x(3)*(1-2*a*Ts-b*Ts) + u(3)*g*Ts*(t_h-x(3)) + b*Ts*t_e, x_max(3)), x_min(3));
end
