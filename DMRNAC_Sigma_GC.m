% Controle ROBUSTO NEURO-ADAPTATIVO MRAC para altitude de foguete
% Tipo: Direto 
% Regra: Lyapunov
% Método: Modelo de referência
% Modificação sigma + Recuperação de desempenho
%----------------------------------------------------------

clear
clc

% ----- Sistema em Espaço de Estados
A = [0 1;
    0 -0.1190];
B = [0;
    373.3151];
C = [1 0];
D = 0;

sys = ss(A,B,C,D);
display(sys);

% Condições iniciais
x = [0; 0];
xm = [0; 0];
xmi = [0; 0];
Psi = [0; 0]; % Vetor de estado do governador de comando

% Controlador Nominal
% Q_lqr = [1 0;
%         0 .1];
% R_lqr = 0.1;
% K1 = lqr(sys,Q_lqr,R_lqr);

K1 = acker(A,B,[-1 -2]); % Ganho de realimentação de estados
K2 = -inv(C*inv(A-B*K1)*B); % Ganho de alimentação direta da referência

% Taxa de aprendizagem
gamma = 2; %2

% Modificação sigma
sigma = 500; %500

% Recuperação de desempenho
lambda = .1; %.1

% Condições de Correspondência do Modelo
Am = A-B*K1;
Bm = B*K2;

% Sinal de correção (Recuperação de Desempenho)
G = inv(K2)*inv(B'*B)*B'; % Matriz do sinal governador de comando
Omega = B*inv(B'*B)*B';

% Equação algébrica de Lyapunov
Q = eye(2);
P = lyap(Am',Q); %solução da equação algébrica de Lyapunov

% Condições iniciais para W_hat
n = 500; % Quantidade de neurônios para cada estado do sistema
b = 100; % limite do domínio D (1000 metros de altitude)
W_hat = zeros(2*n+1,1);
y = C*x;

% Parâmetros da RNA
centers = linspace(-b, b, n); % Vetor de centros dos neurônios
Theta = zeros(2*n, 1); % Inicialização do vetor regressor da RNA

% Simulação
ft = 20;
dt = 0.001;
index = 1;

for k = 0:dt:ft
    
    Theta0 = [1; x(2)^2; x(2)^2];
    delta = 1+x(1)+x(2)+x(1)^2+sin(x(1))+cos(x(1))+sin(x(2))+cos(x(2)); 
    % delta = 1+x(1)^2+sin(x(1))*x(1)^3+cos(x(1))*x(1)^4+x(2)*x(1)+x(2)^2+abs(x(1))^2+x(2)*abs(x(1))^5; % Incerteza

    % Referência
    % if k <= 10
    %     r = 1;
    % end
    % if k > 10
    %     r = -1;
    % end
    if k < 2
        r = 0;
    end
    if k >= 2
        r = 2;
    end
    if k >= 6
        r = -2;
    end
    if k >= 10
        r = 2;
    end
    if k >= 14
        r = -2;
    end

    % if k <= 10
    %     r = 1;
    % end
    % if k > 10
    %     r = 2;
    % end
    % if k > 20
    %     r = 3;
    % end
    % if k > 30
    %     r = 4;
    % end
    % if k > 40
    %     r = 5;
    % end
    % if k > 50
    %     r = 10;
    % end
    % if k > 60
    %     r = 10;
    % end
    % if k > 61
    %     r = -10;
    % end
    % if k > 62
    %     r = 10;
    % end
    % if k > 63
    %     r = 20;
    % end
    % if k > 64
    %     r = 30;
    % end
    % if k > 65
    %     r = 40;
    % end
    
    % Montagem do vetor regressor da RNA
    for i = 1:n
        Theta(i) = exp(-0.25*(abs(x(1)-centers(i)))^2); % RBFs
        Theta(i+n) = exp(-0.25*(abs(x(2)-centers(i)))^2);
    end
    Theta(2*n+1) = 1; % Bias
    
    % Sinal de correção para recuperação
    Psi = Psi + dt*(-lambda*(Psi-(x - xm))); % Dinâmica do governador de comando
    v = lambda*Psi + (Am-lambda*eye(2))*((x-xm)); % Sinal do governador de comando
    
    % Sinal de controle
    u = -K1*x + K2*(r+G*v) - W_hat'*Theta;
    e = x-xmi;
    W_til = delta - W_hat'*Theta;
    V =  e' * P * e + W_til' * inv(gamma) * W_til;
    dV = 2*e'*P*Am*e;
    xm = xm + dt*(Am*xm + Bm*r + Omega*v); % modelo de referência
    xmi = xmi + dt*(Am*xmi + Bm*r); % apenas para plotagem do modelo ideal
    W_hat = W_hat + dt*(gamma*(Theta*(x-xm)'*P*B - sigma*W_hat)); % Atualização dos pesos (adaptação)
    
    % Sistema Atual
    x = x + dt*(A*x + B*(u + delta));
    y = C*x;

    % display(x)
    % display(xm)
    % display(xmi)
    % display(Psi)
    % display(v)
    % display(u)
    % display(W_hat)
    % display(delta)
    
    % Gravação dos dados
    delta_rec(index,1) = delta;
    w_theta_rec(index,1) = W_hat'*Theta;
    r_rec(index,1) = r;
    xm_rec(index,1:2) = xm;
    xmi_rec(index, 1:2) = xmi;
    x_rec(index,1:2) = x;
    u_rec(index,1) = u;
    t_rec(index,1) = k;
    e_rec(index,1:2) = e;
    e_x1_rec(index,1) = e(1);
    w_til_rec(index,1) = W_til;
    v_rec(index,1) = V;
    dv_rec(index,1) = dV;
    e_rbf_rec(index,1) = delta - W_hat'*Theta;
    index = index + 1;
end

% Plote a superfície 3D da função de Lyapunov
% [X,Y] = meshgrid(e_rec,w_til_rec);
% Z = X.^2 + Y.^2;

% figure;
% mesh(X,Y,Z)
% xlabel('X');
% ylabel('Y');
% zlabel('Z');
% surf(e_x1_rec, w_til_rec, v_rec);
% xlabel('e(t)');
% ylabel('W_til');
% zlabel('V(x)');
% title('Função de Lyapunov');

% Gráfico da função candidata de Lyapunov
figure;
plot3(e_rec(:,1), w_til_rec, v_rec, 'g^', 'LineWidth', 1);
grid on;
xlabel('$e(t)$','interpreter','latex');
ylabel('$\widetilde{W}(t)$','interpreter','latex');
zlabel('$V(e(t),\widetilde{W}(t))$','interpreter','latex');
%title('Função candidata de Lyapunov');

% Gráfico da derivada da função candidata de Lyapunov
figure;
plot3(e_rec(:,1), w_til_rec, dv_rec, 'bo', 'LineWidth', 1);
grid on;
xlabel('$e(t)$','interpreter','latex');
ylabel('$\widetilde{W}(t)$','interpreter','latex');
zlabel('$\dot{V}(e(t),\widetilde{W}(t))$','interpreter','latex');
%title('Derivada da função candidata de Lyapunov');

% Plot
figure;
subplot(4,1,1); hold on; box on; grid;
%title('DMNAC, mod-$\sigma$, sinal GC','fontsize',16,'interpreter','latex');
p0 = plot(t_rec,r_rec,'r:');
set(p0,'linewidth',4);
p1 = plot(t_rec, xmi_rec(:,1), 'c-');
set(p1, 'linewidth', 3);
% p1 = plot(t_rec,xm_rec(:,1),'b--');
% set(p1,'linewidth',3);
p2 = plot(t_rec,x_rec(:,1),'k-');
set(p2,'linewidth',2);
xlabel('$t$ (s)','fontsize',10,'interpreter','latex');
ylabel('$x_1$ (m)','fontsize',16,'interpreter','latex');
legend('$r$','$x(1)_m$','x(1)','fontsize',10,'interpreter','latex');
axis tight;

subplot(4,1,2); hold on; box on; grid;
p1 = plot(t_rec,xmi_rec(:,2),'c-');
set(p1,'linewidth',3);
p2 = plot(t_rec,x_rec(:,2),'k-');
set(p2,'linewidth',2);
xlabel('$t$ (s)','fontsize',10,'interpreter','latex');
ylabel('$x_2$ (m/s)','fontsize',16,'interpreter','latex');
legend('$x(2)_m$','$x(2)$','fontsize',10,'interpreter','latex');
axis tight;

subplot(4,1,3); hold on; box on; grid;
p3 = plot(t_rec,u_rec,'b-');
set(p3,'linewidth',3);
xlabel('$t$ (s)','fontsize',10,'interpreter','latex');
ylabel('$u(t)$','fontsize',16,'interpreter','latex');
legend('$u(t)$','fontsize',10,'interpreter','latex');
axis tight;

subplot(4,1,4); hold on; box on; grid;
plot(t_rec,delta_rec,'g-','linewidth',3); hold on; box on; grid;
plot(t_rec,w_theta_rec,'r--','linewidth',3); grid;
xlabel('$t$ (s)','fontsize',10,'interpreter','latex');
ylabel('$\Delta(x)$','fontsize',16,'interpreter','latex');
legend('$\Delta(x)$','$\hat{W}^T\Theta(x)$','fontsize',10,'interpreter','latex');
%axis tight;

% Erro de Estimação da Rede RBF

figure;
plot(t_rec,e_rbf_rec,'m-','linewidth',3); box on; grid on;
%title('Erro de Estimação da Rede RBF','fontsize',12,'interpreter','latex');
xlabel('$t$ (s)','fontsize',10,'interpreter','latex');
ylabel('$e_{RBF}$','fontsize',16,'interpreter','latex');
legend('$e_{RBF}$','fontsize',10,'interpreter','latex');
axis tight;

% Erro de Rastreamento do Sistema

figure;
subplot(2,1,1); hold on; box on; grid;
%title('Erro de Rastreamento do Modelo de Referência','fontsize',12,'interpreter','latex');
p0 = plot(t_rec,e_rec(:,1),'r-');
set(p0,'linewidth',3);
xlabel('$t$ (s)','fontsize',10,'interpreter','latex');
ylabel('$e_{x_1}$ (m)','fontsize',16,'interpreter','latex');
legend('$e_{x_1}$','fontsize',10,'interpreter','latex');
axis tight;

subplot(2,1,2); hold on; box on; grid;
p0 = plot(t_rec,e_rec(:,2),'b-');
set(p0,'linewidth',3);
xlabel('$t$ (s)','fontsize',10,'interpreter','latex');
ylabel('$e_{x_2}$ (m/s)','fontsize',16,'interpreter','latex');
legend('$e_{x_2}$','fontsize',10,'interpreter','latex');
axis tight;

% Calculando as métricas de regressão

% Dados observados (incerteza)
y_real = delta_rec;  % Incerteza observada (dados reais)

% Dados previstos pela rede neural RBF
y_previsto = w_theta_rec;  % Valores previstos pela rede neural RBF

% Métricas de desempenho

% Erro Médio Absoluto (Mean Absolute Error - MAE)
MAE = mean(abs(y_real - y_previsto));

% Erro Quadrático Médio (Mean Squared Error - MSE)
MSE = mean((y_real - y_previsto).^2);

% Raiz do Erro Quadrático Médio (Root Mean Squared Error - RMSE)
RMSE = sqrt(MSE);

% Coeficiente de Determinação (R-squared)
R_squared = 1 - sum((y_real - y_previsto).^2) / sum((y_real - mean(y_real)).^2);

% Erro Percentual Absoluto Médio (Mean Absolute Percentage Error - MAPE)
MAPE = mean(abs((y_real - y_previsto) ./ y_real)) * 100;

% Exibindo as métricas
disp(['(MAE): ', num2str(MAE)]);
disp(['(MSE): ', num2str(MSE)]);
disp(['(RMSE): ', num2str(RMSE)]);
disp(['(R-squared): ', num2str(R_squared)]);
disp(['(MAPE): ', num2str(MAPE)]);

% Gráficos das métricas de regressão ao longo do tempo

% Definindo o intervalo de tempo de 0 a 5 segundos com intervalo de 0.001 segundos
tempo_intervalo = 0:0.001:0.4;

% Encontrando os índices correspondentes ao intervalo de tempo desejado
indice_inicio = find(t_rec >= 0, 1);
indice_fim = find(t_rec >= 0.4, 1);

% Selecionando os dados correspondentes ao intervalo de tempo desejado
tempo_selecionado = t_rec(indice_inicio:indice_fim);
y_real_selecionado = delta_rec(indice_inicio:indice_fim);
y_previsto_selecionado = w_theta_rec(indice_inicio:indice_fim);

% Criando o subplot
figure;

% Erro Médio Absoluto (Mean Absolute Error - MAE)
subplot(5,1,1);
plot(tempo_selecionado, abs(y_real_selecionado - y_previsto_selecionado), 'b-', 'LineWidth', 2);
xlabel('t (s)');
ylabel('MAE');
title('(MAE)');
grid on;

% Erro Quadrático Médio (Mean Squared Error - MSE)
subplot(5,1,2);
plot(tempo_selecionado, (y_real_selecionado - y_previsto_selecionado).^2, 'r-', 'LineWidth', 2);
xlabel('t (s)');
ylabel('MSE');
title('(MSE)');
grid on;

% Raiz do Erro Quadrático Médio (Root Mean Squared Error - RMSE)
subplot(5,1,3);
plot(tempo_selecionado, sqrt((y_real_selecionado - y_previsto_selecionado).^2), 'g-', 'LineWidth', 2);
xlabel('t (s)');
ylabel('RMSE');
title('(RMSE)');
grid on;

% Coeficiente de Determinação (R-squared)
subplot(5,1,4);
plot(tempo_selecionado, R_squared * ones(size(tempo_selecionado)), 'm-', 'LineWidth', 2);
xlabel('t (s)');
ylabel('$R^2$','fontsize',12,'interpreter','latex');
title('(R-squared)');
grid on;

% Erro Percentual Absoluto Médio (Mean Absolute Percentage Error - MAPE)
subplot(5,1,5);
plot(tempo_selecionado, abs((y_real_selecionado - y_previsto_selecionado) ./ y_real_selecionado) * 100, 'c-', 'LineWidth', 2);
xlabel('t (s)');
ylabel('MAPE');
title('(MAPE)');
grid on;

