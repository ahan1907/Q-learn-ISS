clear; 
clc; 

% Define the vertices of the desired temparature domain
goal_vertices = [2 0.02; 3 0.03];
Treshold=0.0;

% Define state and action space dimensions
stateDim = 2;
actionDim = 1;

% Define state and action space boundaries
stateMin = [0; 0];
stateMax = [5; 0.05];
actionMin = 0;
actionMax = 5;

% Discretize state and action spaces (adjust based on your needs)
n_x = 80;
n_modes = 100;
samplingTime = 0.01;

stateGrid = cell(stateDim, 1);
actionGrid = cell(actionDim, 1);
for i = 1:stateDim
  stateGrid{i} = linspace(stateMin(i), stateMax(i), n_x);
end
for i = 1:actionDim
  actionGrid{i} = linspace(actionMin(i), actionMax(i), n_modes);
end

% Define Q-table
Q = zeros(n_x^stateDim, n_modes^actionDim);

% Set training parameters
maxEpisodes = 5000;
maxStepsPerEpisode = 6000;
learningRate = 0.9;
discountFactor = 0.99;
epsilon = 0.99; % Exploration rate
goal_reward = 0;

% Training loop
for i = 1:maxEpisodes
  % Reset environment (assuming initial state is 0)
  state = [4, 0.012];
  
  for t = 1:maxStepsPerEpisode
    % Discretize state
    stateLine = discretizeState(state, stateGrid, stateDim, n_x);
    
    % Choose action based on epsilon-greedy exploration
    if rand < epsilon
      action = randi(n_modes, 1);
    else
      [~, action] = max(Q(stateLine, :));
    end
    
    % Convert the linear index to row index
    row_idx = action; 

    % Convert action from index to continuous value
    continuousAction = actionGrid{1}(row_idx);
    
    % Take action and observe reward (replace with your reward function)
    nextX = update_environment(state, continuousAction, 0.1); % Ts = 0.1 assumed

    % Example reward
    % Check if the temparature is inside the desired domain
    is_inside_goal = (nextX(1) >= goal_vertices(1,1)) && (nextX(1) <= goal_vertices(2,1)) && ...
                     (nextX(2) >= goal_vertices(1,2)) && (nextX(2) <= goal_vertices(2,2));

    if is_inside_goal
        reward = 0;
    else
        reward = -1; 
    end


    % Discretize next state
    nextXLine = discretizeState(nextX, stateGrid, stateDim, n_x);

    % Update Q-value using Q-learning update rule
    qValue = Q(stateLine, action);
    maxQnext = max(Q(nextXLine, :));
    newQValue = qValue + learningRate * (reward + discountFactor * maxQnext - qValue);
    Q(stateLine, action) = newQValue;

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
optimalPolicy = zeros(n_x^stateDim, actionDim);
for stateLine = 1:n_x^stateDim
    % Find the action with the highest Q-value for the current state
    [~, optimalAction] = max(Q(stateLine, :));

    % Convert the linear index to row index
    row_idx = optimalAction; 
    
    % Convert action from index to continuous value
    continuousAction = actionGrid{1}(row_idx);
    
    % Store the optimal action for the current state
    optimalPolicy(stateLine, :) = continuousAction;
end

Visualization
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
    x_max = [5; 0.05];
    
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
