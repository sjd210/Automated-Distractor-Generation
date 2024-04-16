import torch
import torch.nn.functional as functional

class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.loss = torch.nn.MSELoss()
        self.register_buffer('target', torch.tensor(0))

    def __call__(self, input):
        return self.loss(input, self.target.expand_as(input))

# ==========

class Generator(torch.nn.Module):
    def __init__(self, G_v1, G_v2, temperature):
        super(Generator, self).__init__()
        self.G_v1 = G_v1
        self.G_v2 = G_v2
        self.temperature = temperature

    def forward(self, samples, indexes, reward, importance):
        summedScores = torch.gather(functional.softmax(self.get_score(samples).view(1, -1), -1).view(-1), 0, indexes)
        return -torch.mean(torch.log(summedScores) * reward.view(-1) * importance.view(-1))

    def get_score(self, samples):
        qss = samples[:, 0].tolist()
        ass = samples[:, 1].tolist()
        dss = samples[:, 3].tolist()

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        output1 = cos(torch.tensor(qss).cuda().float(), torch.tensor(dss).cuda().float()).cuda()
        output2 = cos(torch.tensor(ass).cuda().float(), torch.tensor(dss).cuda().float()).cuda()

        self.score = (torch.matmul(output1, self.G_v1) + torch.matmul(output2, self.G_v2))/self.temperature
        return self.score

# ==========

class Discriminator(torch.nn.Module):
    def __init__(self, D_v1, D_v2):
        super(Discriminator, self).__init__()
        self.D_v1 = D_v1
        self.D_v2 = D_v2

    def forward(self, samples, labels):
        return torch.mean(functional.binary_cross_entropy_with_logits(
                            self.get_score(samples).view(-1, 1).float(), 
                            labels.view(-1, 1).float())) 

    def get_score(self, samples):
        qss = samples[:, 0].tolist()
        ass = samples[:, 1].tolist()
        dss = samples[:, 2].tolist()

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        output1 = cos(torch.tensor(qss).cuda().float(), torch.tensor(dss).cuda().float()).cuda()
        output2 = cos(torch.tensor(ass).cuda().float(), torch.tensor(dss).cuda().float()).cuda()

        self.score = torch.matmul(output1, self.D_v1) + torch.matmul(output2, self.D_v2)
        return self.score 

    def get_reward(self, samples):
        negs = [[q,a,n] for [q,a,d,n] in samples]
        dists = [[q,a,d] for [q,a,d,n] in samples]
        return self.get_score(torch.tensor(dists).cuda()) - self.get_score(torch.tensor(negs).cuda())  # CHANGED THE ORDER HERE
