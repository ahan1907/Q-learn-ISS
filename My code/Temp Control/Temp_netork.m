clc
clear

% Compute intermediate variables
a = 1/20;
b = 1/200;
g = 1/100;
Ts = 0.2;
t_e = 0;
t_h = 50;

N = 7000;
x = zeros(N,3);
x(1,:) = [10 -10 15];

u = [0.1 0 0.3];

for i = 1:length(x)-1
    x(i+1,1) = x(i,1)*(1-2*a*Ts-b*Ts) + x(i,2)*a*Ts + x(i,3)*a*Ts + u(1)*g*Ts*(t_h-x(1));
    x(i+1,2) = x(i,1)*a*Ts + x(i,2)*(1-2*a*Ts-b*Ts) + x(i,3)*a*Ts + u(2)*g*Ts*(t_h-x(2));
    x(i+1,3) = x(i,1)*a*Ts + x(i,2)*a*Ts + x(i,3)*(1-2*a*Ts-b*Ts) + u(3)*g*Ts*(t_h-x(3));
end

t = linspace(0,10,N);

figure(1)
stairs(t, x(:,1))
hold on
stairs(t, x(:,2))
hold on
stairs(t, x(:,3))
grid on
legend('R1', 'R2', 'R3')
hold off