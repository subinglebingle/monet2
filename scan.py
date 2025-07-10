#scan model 정의
class BVAE(nn.Module): #학습 시 scan의 이미지 처리 단(beta VAE)
    def __init__(self, input_size, hidden_size, output_size):
        super(BVAE, self).__init__()
        self.output_size = output_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size * 2)  # 인코더가 뽑아야 할 output이 두개 (mu(평균), logvar(분산))
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoder_output = self.encoder(x)
        mu, logvar = encoder_output[:, :self.output_size], encoder_output[:, self.output_size:]
        z = mu + torch.randn_like(torch.exp(0.5 * logvar)) #sigma=root(var)=root(exp(log(var))) ->양수로 만들기 위해서 =exp(0.5*log(var))
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def compute_loss(self, x, recon_x, mu, logvar, dae, beta):
        z_d = dae.encode(x)
        recon_z_d = dae.encode(recon_x)
        delta = z_d - recon_z_d
        L2_loss = 0.5 * torch.sum(delta * delta)
        KLD_loss = -0.5 * beta * torch.sum(1 + logvar - mu * mu - logvar.exp())
        return L2_loss, KLD_loss

class DAE(nn.Module): #denoising autoencoder, noise를 제거하는 VAE로 Beta VAE 학습 과정에서 오차 보정을 위해 사용.
    def __init__(self, input_size, hidden_size, output_size):
        super(DAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ELU(inplace=False),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(inplace=False),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(inplace=False),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(inplace=False)
        )

        self.fc1 = nn.Linear(hidden_size, output_size)
        self.fc2 = nn.Linear(output_size, hidden_size)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(inplace=False),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(inplace=False),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(inplace=False),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

        self.elu = nn.ELU(inplace=False)
        self.tanh = nn.Tanh()

    def encode(self, x):
        h = self.encoder(x)
        z = self.tanh(self.fc1(h))
        return z

    def decode(self, z):
        h = self.elu(self.fc2(z))
        out = self.decoder(h)
        return out

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out

    def compute_loss(self, x_org, x_out):
        delta = x_org - x_out
        reconstr_loss = 0.5 * torch.sum(delta * delta)
        return reconstr_loss

class SCAN(nn.Module): #학습시 라벨 입력 단이자 scan 모델에서 최종적으로 학습시켜야 하는 것. 나중에 추론 및 생성에서 이 부분 이용.
    def __init__(self, input_size, hidden_size, output_size):
        super(SCAN, self).__init__()
        self.output_size = output_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size * 2)  # 인코더가 뽑아야 할 output이 두개
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, y):
        encoder_output = self.encoder(y)
        mu, logvar = encoder_output[:, :self.output_size], encoder_output[:, self.output_size:]
        logvar = torch.clamp(logvar, min=-10, max=10) #NaN값 나오는 이슈 방지
        z = mu + torch.randn_like(torch.exp(0.5 * logvar))
        out_y = self.decoder(z)
        return z, out_y, mu, logvar

    def compute_loss(self, x, y, target, mu, logvar, x_mu, x_logvar, lambd=10.0):
        reconstr_loss = nn.BCELoss()(y, target) #loss function 바꿔봄
        KLD_loss_1 = -0.5 * torch.sum(1 + logvar - mu * mu - logvar.exp())
        KLD_loss_2 = -lambd * self._kl(x_mu, x_logvar, mu, logvar) #.................beta는 설정안하나????
        return reconstr_loss, KLD_loss_1, KLD_loss_2

    def _kl(self, mu1, logvar1, mu2, logvar2):
        mu = mu1 - mu2
        return torch.sum(0.5 * (logvar2 - logvar1 + (logvar1.exp() - logvar2.exp()) + mu * mu / logvar2.exp() - 1))

class Recombinator(nn.Module): #recombinator 최종적으로 필요한 걸로 결론
    def __init__(self, input_size, hidden_size, output_size):
        super(Recombinator, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.conv = nn.Sequential(
            nn.Conv1d(4, 1024, 1), #mu0, logvar0, mu1, logvar1
            nn.ReLU(),
            nn.Conv1d(1024, 6, 1) #.........앞에 3개평균내서 r_mu, 뒤에 3개 평균내서 r_logvar로 쓰던데 왜 3개씩일까
        )

    def recombine(self, mu0, logvar0, mu1, logvar1):
        z_stacked = torch.stack([mu0, mu1, logvar0, logvar1], 1)
        return z_stacked

    def reparameterize(self, mu, logvar): #주어진 평균과 분산을 이용하여 정규분포에서 샘플링된 latent 벡터 z 생성
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def forward(self, mu0, logvar0, mu1, logvar1):
        z_stacked = self.recombine(mu0, logvar0, mu1, logvar1)
        h = self.conv(z_stacked)
        mu, logvar = torch.split(h, 3, 1)
        r_mu = torch.mean(mu, 1)  # 평균 사용으로 결합
        r_logvar = torch.mean(logvar, 1)  # 평균 사용으로 결합
        r_z = self.reparameterize(r_mu, r_logvar)
        return r_z, r_mu, r_logvar

    def compute_loss(self, r_mu, r_logvar, x_mu, x_logvar, y_mu, y_logvar):
        symbol_loss = self._kl(y_mu, y_logvar, r_mu, r_logvar) #...forward kl 아마도.,?
        return _, symbol_loss

    def _kl(self, mu1, logvar1, mu2, logvar2):
        mu = mu1 - mu2
        return torch.sum(0.5 * (logvar2 - logvar1 + (logvar1 - logvar2).exp() + torch.mul(mu, mu) / logvar2.exp() - 1))
