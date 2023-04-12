import os, time, pickle, argparse
import pickle as pk
import sys
sys.path.append("../")
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from scipy.stats import beta
from sklearn.linear_model import LinearRegression, LogisticRegression
from nonstationary_contextualbandits.contextualbandits.online import BootstrappedUCB, BootstrappedTS, LogisticUCB, \
            SeparateClassifiers, EpsilonGreedy, AdaptiveGreedy, ExploreFirst, \
            ActiveExplorer, SoftmaxExplorer, LinUCB
from copy import deepcopy
torch.set_printoptions(threshold=10000)
np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser(description='RSAutoML')
parser.add_argument('--Train_Method', type=str, default='AutoML', help='options: AutoML, Supervised')
parser.add_argument('--Val_Type', type=str, default='last_batch', help='options: last_batch, last_random')
parser.add_argument('--Loss_Type', type=str, default='MSE_sigmoid', help='options: MSE_sigmoid   MSE_no_sigmoid  BCEWithLogitsLoss   CrossEntropyLoss')
parser.add_argument('--Data_Set', type=str, default='ml-20m', help='options: ml-20m ml-latest')
parser.add_argument('--Dy_Emb_Num', type=int, default=2, help='options: 1, 2')
parser.add_argument('--Reward_Base', type=str, default=None, help='options: None, last_loss, ave_loss')
parser.add_argument('--last_num', type=int, default=5, help='options: 1, 2')
parser.add_argument('--alpha', type=float, default=0.8)
parser.add_argument('--GPU', type=str, default='1', help='options: 0, 1, 2, 3, 4, 5, 6, 7')

args = parser.parse_args()
GPU = args.GPU
Model_Gpu  = torch.cuda.is_available()
device     = torch.device('cuda:{}'.format(GPU) if Model_Gpu else 'cpu')
DATA_PATH  = '../data'
DATA_SET   = args.Data_Set
Batch_Size = 500     # batch size
LR_model   = 0.001   # learning rate
LR_darts   = 0.0001  # learning rate
Epoch      = 1       # train epoch
Beta_Beta  = 20      # beta for Beta distribution
H_alpha    = 0       # for nn.KLDivLoss 0.001
MAX_NORM = 1

if DATA_SET == 'ml-20m':
    Train_Size   = 16000000      # training dataset size
elif DATA_SET == 'ml-latest':
    Train_Size = 22000000  # training dataset size
elif DATA_SET == 'Amazon_Books':
    Train_Size = 2400000
elif DATA_SET == 'Amazon_CD':
    Train_Size = 500000

if DATA_SET == 'ml-20m':
    Test_Size   = 3800000      # testing dataset size
elif DATA_SET == 'ml-latest':
    Test_Size = 5000000  # testing dataset size
elif DATA_SET == 'Amazon_Books':
    Test_Size = 610555
elif DATA_SET == 'Amazon_CD':
    Test_Size = 143455
    
#Emb_Size     = [2, 4, 8, 16, 64, 128]  # 1,2,4,8,16,32,64,128,256,512
#Emb_Size = np.array([2, 4, 8, 16, 64, 96, 128, 192 ])
Emb_Size = np.array([2, 4, 8, 16, 64, 128])
Train_Method = args.Train_Method
#Policy_Type  = args.Policy_Type
Types        = ['Policy0: embedding for popularity',
                'Policy1: embedding for popularity + last_weights',
                'Policy2: embedding for popularity + last_weights + last_loss',
                'Policy3: popularity one_hot',
                'Policy4: popularity one_hot + last_weights',
                'Policy5: popularity one_hot + last_weights  + last_loss']
Val_Type     = args.Val_Type

Dy_Emb_Num   = args.Dy_Emb_Num       # dynamic num of embedding to adjust, 1 for user, 2 for user & movie
Reward_Base  = args.Reward_Base
last_num = args.last_num
Loss_Type    = args.Loss_Type
ControllerLoss = nn.CrossEntropyLoss(reduce=False)
base_algorithm = LogisticRegression(solver='lbfgs', warm_start=True)
threshold = 5

def load_data_Amazon():
    train_features, test_features, train_target, test_target \
        = pickle.load(open('{}/{}_TrainTest_{}_{}.data'.format(DATA_PATH, DATA_SET, Train_Size, Output_Dim), mode='rb'))
    test_features, test_target = test_features[:Test_Size], test_target[:Test_Size]
    train_feature_data = pd.DataFrame(train_features, columns=['userID', 'movieID', 'user_frequency', 'movie_frequency'])
    test_feature_data = pd.DataFrame(test_features, columns=['userID', 'movieID', 'user_frequency', 'movie_frequency'])
    User_Num = max(train_feature_data['userID'].max() + 1, test_feature_data['userID'].max() + 1)  # 138494
    Movie_Num = max(train_feature_data['movieID'].max() + 1, test_feature_data['movieID'].max() + 1)  # 131263

    max_user_popularity = max(train_feature_data['user_frequency'].max()+1, test_feature_data['user_frequency'].max()+1)
    max_movie_popularity = max(train_feature_data['movie_frequency'].max() + 1, test_feature_data['movie_frequency'].max() + 1)
    return train_features, test_features, train_target, test_target,  \
           train_feature_data, test_feature_data, len(train_features), len(test_features), \
           User_Num, Movie_Num, max_user_popularity, max_movie_popularity

def load_data():
    train_features, test_features, train_target, test_target \
        = pickle.load(open('{}/{}_TrainTest_{}_{}.data'.format(DATA_PATH, DATA_SET, Train_Size, Output_Dim), mode='rb'))
    test_features, test_target = test_features[:Test_Size], test_target[:Test_Size]
    genome_scores_dict = pickle.load(open('./{}/{}_GenomeScoresDict.data'.format(DATA_PATH, DATA_SET), mode='rb'))
    train_feature_data = pd.DataFrame(train_features, columns=['userId', 'movieId', 'user_frequency', 'movie_frequency'])
    test_feature_data = pd.DataFrame(test_features, columns=['userId', 'movieId', 'user_frequency', 'movie_frequency'])
    User_Num = max(train_feature_data['userId'].max() + 1, test_feature_data['userId'].max() + 1)  # 138494
    Movie_Num = max(train_feature_data['movieId'].max() + 1, test_feature_data['movieId'].max() + 1)  # 131263
    max_user_popularity = max(train_feature_data['user_frequency'].max()+1, test_feature_data['user_frequency'].max()+1)
    max_movie_popularity = max(train_feature_data['movie_frequency'].max() + 1, test_feature_data['movie_frequency'].max() + 1)
    return train_features, test_features, train_target, test_target, genome_scores_dict, \
           train_feature_data, test_feature_data, len(train_features), len(test_features), \
           User_Num, Movie_Num, max_user_popularity, max_movie_popularity

def Batch_Losses(Loss_Type, prediction, target):
    if Loss_Type == 'MSE_sigmoid':
        return nn.MSELoss(reduction='none')(nn.Sigmoid()(prediction), target)
    elif Loss_Type == 'MSE_no_sigmoid':
        return nn.MSELoss(reduction='none')(prediction, target)
    elif Loss_Type == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss(reduction='none')(prediction, target)
    elif Loss_Type == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss(reduction='none')(prediction, target)
    else:
        print('No such Loss_Type.')


def Batch_Accuracies(Loss_Type, prediction, target):
    with torch.no_grad():
        if Loss_Type == 'MSE_sigmoid':
            predicted = 1 * (torch.sigmoid(prediction).data > 0.5)
        elif Loss_Type == 'MSE_no_sigmoid':
            predicted = 1 * (prediction > 0.5)
        elif Loss_Type == 'BCEWithLogitsLoss':
            predicted = 1 * (torch.sigmoid(prediction).data > 0.5)
        elif Loss_Type == 'CrossEntropyLoss':
            _, predicted = torch.max(prediction, 1)
        else:
            print('No such Loss_Type.')

        Batch_Accuracies = 1 * (predicted == target)
        Batch_Accuracies = list(Batch_Accuracies.detach().cpu().numpy())
        return Batch_Accuracies


def Beta(length, popularity, be=10):
    x = [i/length for i in range(length+1)]
    cdfs = [beta.cdf(x[i+1], popularity, be) - beta.cdf(x[i], popularity, be) for i in range(length)]
    return cdfs

def hook_backward_fn(module, grad_input, grad_output):
    grad = grad_input
    return grad

class Network_u(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network_u, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))

class Network_Arm(nn.Module):
    def __init__(self, Setting_Popularity, p=0.5):
        super(Network_Arm, self).__init__()
        self.p = p
        self.transform_input_length = Setting_Popularity[1]
        self.emb_popularity = nn.Embedding(num_embeddings=Setting_Popularity[0], embedding_dim=Setting_Popularity[1])
        self.batch_norm = nn.BatchNorm1d(Setting_Popularity[1])
        self.transform = nn.Sequential(
            nn.Linear(self.transform_input_length, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Linear(512, 1))
        self.emb_popularity.register_backward_hook(hook_backward_fn)
        self.batch_norm.register_backward_hook(hook_backward_fn)
        self.transform.register_backward_hook(hook_backward_fn)


    def forward(self, popularity):
        #emb_popularity = self.emb_popularity(popularity)
        #transformed_emb_popularity = self.batch_norm(emb_popularity)
        return self.transform(nn.functional.dropout(self.batch_norm(self.emb_popularity(popularity)), p=self.p))


class neural_bandit():
    def __init__(self, setting_popularity, number_arm=6, lr=0.01, lamdba=1.0, nu=0.01):
        self.context_list = []
        self.reward = []
        self.lr = lr
        self.lamdba = lamdba
        self.number_arm = number_arm
        self.setting_popularity = setting_popularity
        self.nu = nu
        self.arm_funcs = {}
        self.Arm = {}
        self.inip = {}
        self.optimizer = {}

        for i in range(self.number_arm):
            self.arm_funcs[i] = Network_Arm(Setting_Popularity=self.setting_popularity).to(device)
            #self.arm_funcs[i] = Network_u(dim=1).to(device)
            self.optimizer[i] = torch.optim.SGD(self.arm_funcs[i].parameters(), lr=self.lr)
            self.inip[i] = self.arm_funcs[i].state_dict()
            total_param = sum(p.numel() for p in self.arm_funcs[i].parameters() if p.requires_grad)
            self.Arm[i] = lamdba * torch.ones((total_param,)).to(device)

        self.contexts = defaultdict(list)
        self.rewards = defaultdict(list)

    def choose_arm(self, context):
        gradient_list = torch.tensor([]).to(device)
        ucb_list = torch.tensor([]).to(device)
        res_list = torch.tensor([]).to(device)
        sample_rs = torch.tensor([]).to(device)
        #for index in range(len(context)):
        for i in range(self.number_arm):
            res = self.arm_funcs[i](context)
            loss = torch.sum(res, dim=0)/len(res)
            self.arm_funcs[i].zero_grad()
            loss.requires_grad_(True)
            loss.backward(retain_graph=True)
            #gradient = torch.cat([p.grad.flatten().detach() for p in self.arm_funcs[i].parameters()])
            gradient = torch.cat([p.grad.flatten().detach() if p.grad is not None else torch.tensor([0]) for p in self.arm_funcs[i].parameters()]).to(device)
            gradient_list = torch.cat((gradient_list, gradient))
            sigma2 = self.lamdba * self.nu * gradient * gradient 
            sigma = torch.sqrt(torch.sum(sigma2))

            sample_r_batch = torch.tensor([]).to(device)
            for index in range(len(context)):
                sample_r = res[index] + sigma
                sample_r_batch = torch.cat((sample_r_batch, sample_r), 0)
            sample_r_batch= torch.reshape(sample_r_batch, (-1, 1))
            sample_rs = torch.cat((sample_rs, sample_r_batch), 1)
            ucb_list = torch.cat((ucb_list, torch.unsqueeze(sigma, 0)))
            res = torch.reshape(res, (-1, 1))
            res_list = torch.cat((res_list, res), 1)
            #self.Arm[i] += gradient_list[i] * gradient_list[i]

        arm = torch.argmax(sample_rs, dim=1)
        return arm, res_list, ucb_list

    def train(self, u):
        optimizer = torch.optim.SGD(self.u_funcs[u].parameters(), lr=self.lr)
        length = len(self.rewards[u])
        index = np.arrange(length)
        np.random.shuffle(index)
        count = 0
        total_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c = self.contexts[u][idx]
                r =  self.rewards[u][idx]
                optimizer.zero_grad()
                loss = (self.u_funcs[u](c.to(device)) - r)**2
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                total_loss += loss.item()
                count += 1
                if count >= 500:
                    return total_loss / count
            if batch_loss / length <= 1e-3:
                return batch_loss / length





        




class RS_MLP_Old(nn.Module):
    def __init__(self, Output_Dim, Dynamic_Emb_Num):
        super(RS_MLP_Old, self).__init__()
        # self.emb_user = nn.Embedding(num_embeddings=User_Num, embedding_dim=sum(Emb_Size))
        # self.emb_movie = nn.Embedding(num_embeddings=Movie_Num, embedding_dim=sum(Emb_Size))
        self.max_norm = MAX_NORM
        self.emb_user = nn.ModuleList(nn.Embedding(num_embeddings=User_Num, embedding_dim=emb_size) for emb_size in Emb_Size)
        self.emb_movie = nn.ModuleList(nn.Embedding(num_embeddings=Movie_Num, embedding_dim=emb_size) for emb_size in Emb_Size)
        # for emb in self.emb_user + self.emb_movie:
        #     emb.to(device)
        self.bn_user = nn.BatchNorm1d(max(Emb_Size))
        self.bn_movie = nn.BatchNorm1d(max(Emb_Size))
        self.W_user = nn.ModuleList([nn.Linear(Emb_Size[i], Emb_Size[i + 1]) for i in range(len(Emb_Size) - 1)])
        self.W_movie = nn.ModuleList([nn.Linear(Emb_Size[i], Emb_Size[i + 1]) for i in range(len(Emb_Size) - 1)])
        self.tanh = nn.Tanh()
        self.movie_transfrom = nn.Sequential(  # nn.BatchNorm1d(1128),
            nn.Linear(1128, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Linear(512, max(Emb_Size)))
        self.den = Dynamic_Emb_Num
        # setattr(self, 'z', 666)

    def forward(self, u_emb_sizes, m_emb_sizes, userId, movieId, movie_vec):
        '''
        u_emb_sizes: (batch_size)
        m_emb_sizes: (batch_size)
        '''

        # u_weight: (batch_size x emb_num)
        # m_weight: (batch_size x emb_num)
        #print('u_emb_sizes', u_emb_sizes)
        #print('u_emb_sizes[0]', u_emb_sizes[0])
        u_weight = nn.functional.one_hot(u_emb_sizes, num_classes=len(Emb_Size))
        if self.den == 2:
            m_weight = nn.functional.one_hot(m_emb_sizes, num_classes=len(Emb_Size))

        user_emb = [self.emb_user[i](userId) for i in range(len(Emb_Size))]
        movie_emb = None if self.den == 1 else [self.emb_movie[i](movieId) for i in range(len(Emb_Size))]

        user_embs = []
        for i in range(len(Emb_Size)):
            temp = user_emb[i]
            for j in range(i, len(Emb_Size) - 1):
                temp = self.W_user[j](temp)
            user_embs.append(temp)

        if self.den == 2:
            movie_embs = []
            for i in range(len(Emb_Size)):
                temp = movie_emb[i]
                for j in range(i, len(Emb_Size) - 1):
                    temp = self.W_movie[j](temp)
                movie_embs.append(temp)

        v_user = sum([torch.reshape(u_weight[:, i], (len(u_weight), -1)) * self.tanh(
            self.bn_user(user_embs[i])) for i in range(len(Emb_Size))])
        v_movie = sum([torch.reshape(m_weight[:, i], (len(m_weight), -1)) * self.tanh(
            self.bn_movie(movie_embs[i])) for i in range(len(Emb_Size))]) if self.den == 2 else self.movie_transfrom(movie_vec)

        dist = torch.cdist(v_user.unsqueeze(1), v_movie.unsqueeze(1)).reshape(-1)
        max_dist = 2 * self.max_norm if self.max_norm is not None else 100
        return max_dist - dist

    def transfer_embedding(self, old_size, new_size, embedding_name='user'):
        if embedding_name == 'user' and new_size == old_size + 1:
            self.emb_user[new_size] = self.W_user[old_size](self.emb_user[old_size])
        elif  embedding_name == 'movie' and new_size == old_size + 1:
            self.emb_movie[new_size] = self.W_movie[old_size](self.emb_movie[old_size])
        else:
            print('No transformation Inquiries')




class RS_MLP_New(nn.Module):
    def __init__(self, Output_Dim, Dynamic_Emb_Num):
        super(RS_MLP_New, self).__init__()
        self.emb_user = nn.ModuleList(nn.Embedding(num_embeddings=User_Num, embedding_dim=emb_size) for emb_size in Emb_Size)
        self.emb_movie = nn.ModuleList(nn.Embedding(num_embeddings=Movie_Num, embedding_dim=emb_size) for emb_size in Emb_Size)
        self.bn_user = nn.BatchNorm1d(max(Emb_Size))
        self.bn_movie = nn.BatchNorm1d(max(Emb_Size))
        self.W_user = nn.ModuleList([nn.Linear(i, max(Emb_Size)) for i in Emb_Size])
        self.W_movie = nn.ModuleList([nn.Linear(i, max(Emb_Size)) for i in Emb_Size])
        self.tanh = nn.Tanh()
        self.movie_transfrom = nn.Sequential(  # nn.BatchNorm1d(1128),
            nn.Linear(1128, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Linear(512, max(Emb_Size)))
        self.transform = nn.Sequential(
            nn.BatchNorm1d(max(Emb_Size) * 2),
            nn.Linear(max(Emb_Size) * 2, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Linear(512, Output_Dim))
        self.den = Dynamic_Emb_Num

    def forward(self, u_emb_sizes, m_emb_sizes, userID, movieID, movie_vec):
        #batch_size x embedding_size of each user
        user_emb = [self.emb_user[u_emb_sizes[i]](userID[i]) for i in range(len(u_emb_sizes))]
        movie_emb = None if self.den == 1 else [self.emb_movie[m_emb_sizes[i]](movieID[i]) for i in range(len(m_emb_sizes))]

        unified_user_emb = torch.tensor([self.W_user[u_emb_sizes[i]](user_emb[i]).detach().cpu().numpy() for i in range(len(u_emb_sizes))]).to(device)
        unified_movie_emb = None if self.den == 1 else torch.tensor([self.W_movie[m_emb_sizes[i]](movie_emb[i]).detach().cpu().numpy() for i in range(len(m_emb_sizes))]).to(device)

        v_user = self.tanh(self.bn_user(unified_user_emb))
        v_movie = self.tanh(self.bn_movie(unified_movie_emb)) if self.den == 2 else self.movie_transform(movie_vec)

        user_movie = torch.cat((v_user, v_movie), 1)
        return self.transform(user_movie)

def update_neural_bandit_controller(index, features, target):
    if Train_Method == 'AutoML' and index > 0:
        if Val_Type == 'last_random':
            val_index = np.random.choice(index, Batch_Size)
            batch_train = features[:index][val_index]
            batch_train_target = target[:index][val_index]
        elif Val_Type == 'last_batch':
            batch_train = features[index - Batch_Size:index]
            batch_train_target = target[index - Batch_Size:index]
        else:
            batch_train = None
            batch_train_target = None
            print('No such Val_Type')

        userId = torch.tensor(batch_train[:, 0], requires_grad=False).to(device)
        movieId = torch.tensor(batch_train[:, 1], requires_grad=False).to(device)
        userPop = torch.tensor(batch_train[:, 2], requires_grad=False).to(device)
        moviePop = torch.tensor(batch_train[:, 3], requires_grad=False).to(device)
        old_uw = torch.tensor(user_weights[batch_train[:, 0]], requires_grad=False).to(device)
        old_mw = torch.tensor(movie_weights[batch_train[:, 1]], requires_grad=False).to(device)
        old_ul = torch.tensor(user_losses[batch_train[:, 0], :], requires_grad=False).to(device)
        old_ml = torch.tensor(movie_losses[batch_train[:, 1], :], requires_grad=False).to(device)
        movie_vec = torch.tensor([genome_scores_dict[str(batch_train[:, 1][i])] for i in range(len(batch_train[:, 1]))],
                                 requires_grad=False).to(device) if Dy_Emb_Num == 1 else None
        if movie_vec is not None:
            movie_vec = movie_vec.to(torch.float32)
        batch_train_target = torch.tensor(batch_train_target,
                                          dtype=torch.int64 if Loss_Type == 'CrossEntropyLoss' else torch.float32,
                                          requires_grad=False).to(device)

        if Reward_Base == 'ave_loss':
            old_utl = torch.tensor(user_total_losses[batch_train[:, 0]], requires_grad=False).to(device)
            old_mtl = torch.tensor(movie_total_losses[batch_train[:, 1]], requires_grad=False).to(device)
            old_uc = torch.tensor(user_count[batch_train[:, 0]], requires_grad=False).to(device)
            old_mc = torch.tensor(movie_count[batch_train[:, 1]], requires_grad=False).to(device)

        new_uw, _, _ = user_policy.choose_arm(context=userPop)
        if Dy_Emb_Num == 2:
            new_mw, _, _ = movie_policy.choose_arm(context=moviePop)
        else:
            new_mw = 0


        with torch.no_grad():
            rating = model(new_uw, new_mw, userId, movieId, movie_vec)
            rating = rating.squeeze(1).squeeze(1).squeeze(1) if Loss_Type == 'CrossEntropyLoss' else ratings.squeeze(1)
            batch_losses = Batch_Losses(Loss_Type, rating, batch_train_target)
            rewards = 1 - batch_losses
        
        if Reward_Base == 'last_loss':
            baseline_u = 1 - old_ul[:, 0]
        elif Reward_Base == 'ave_loss':
            last_num_tensor = torch.Tensor([last_num]).repeat(len(old_uc)).to(device)
            baseline_u = 1 - torch.sum(old_utl, dim=1) / torch.where(old_uc < last_num, old_uc, last_num_tensor)
        rewards_u = rewards - baseline_u
        
        #rewards_u = torch.reshape(rewards_u, (-1, 1))
        rewards_arm_u = [0] * len(Emb_Size)
        loss_u = [0] * len(Emb_Size)
        for i in range(len(Emb_Size)):
            user_policy.optimizer[i].zero_grad()
            rewards_arm_u[i] = rewards_u[new_uw == i]
            if len(rewards_arm_u[i]) > 1 :
                loss_u[i] = torch.sum(torch.square(user_policy.arm_funcs[i](userPop[new_uw == i]) - rewards_arm_u[i])) 
                loss_u[i].backward()
                user_policy.optimizer[i].step()


        if Dy_Emb_Num == 2:
            if Reward_Base == 'last_loss':
                baseline_m = 1 - old_ml[:, 0]
            elif Reward_Base == 'ave_loss':
                last_num_tensor = torch.Tensor([last_num]).repeat(len(old_mc)).to(device)
                baseline_m = 1 - torch.sum(old_mtl, dim=1) / torch.where(old_mc < last_num, old_mc, last_num_tensor)
            rewards_m = rewards - baseline_m
            #rewards_m = torch.reshape(rewards_m, (-1, 1))
            rewards_arm_m = [0] * len(Emb_Size)
            loss_m = [0] * len(Emb_Size)
            for i in range(len(Emb_Size)):
                movie_policy.optimizer[i].zero_grad()
                rewards_arm_m[i] = rewards_m[new_mw == i]
                if len(rewards_arm_m[i]) > 1 :
                    loss_m[i] = torch.sum(torch.square(movie_policy.arm_funcs[i](moviePop[new_mw == i]) - rewards_arm_m[i]))
                    loss_m[i].backward()
                    movie_policy.optimizer[i].step()


def update_ladder_bandit_controller(index, features, target, Setting_Movie_Popularity, Setting_User_Popularity):
    #update the bandit-based user/movie embedding size controller
    if Train_Method == 'AutoML' and index > 0:
        if Val_Type == 'last_random':
            val_index = np.random.choice(index, Batch_Size)
            batch_train = features[:index][val_index]
            batch_train_target = target[:index][val_index]
        elif Val_Type == 'last_batch':
            batch_train = features[index - Batch_Size:index]
            batch_train_target = target[index - Batch_Size:index]
        else:
            batch_train = None
            batch_train_target = None
            print('No such Val_Type')

        userId = torch.tensor(batch_train[:, 0], requires_grad=False).to(device)
        movieId = torch.tensor(batch_train[:, 1], requires_grad=False).to(device)
        userPop = torch.tensor(batch_train[:, 2], requires_grad=False).to(device)
        moviePop = torch.tensor(batch_train[:, 3], requires_grad=False).to(device)
        old_uw = torch.tensor(user_weights[batch_train[:, 0]], requires_grad=False).to(device)
        old_mw = torch.tensor(movie_weights[batch_train[:, 1]], requires_grad=False).to(device)
        old_ul = torch.tensor(user_losses[batch_train[:, 0], :], requires_grad=False).to(device)
        old_ml = torch.tensor(movie_losses[batch_train[:, 1], :], requires_grad=False).to(device)
        movie_vec = torch.tensor([genome_scores_dict[str(batch_train[:, 1][i])] for i in range(len(batch_train[:, 1]))],
                                 requires_grad=False).to(device) if Dy_Emb_Num == 1 else None

        if movie_vec is not None:
            movie_vec = movie_vec.to(torch.float32)
        batch_train_target = torch.tensor(batch_train_target, dtype=torch.int64 if Loss_Type == 'CrossEntropyLoss' else torch.float32, requires_grad=False).to(device)
        if Reward_Base == 'ave_loss':
            old_utl = torch.tensor(user_total_losses[batch_train[:, 0]], requires_grad=False).to(device)
            old_mtl = torch.tensor(movie_total_losses[batch_train[:, 0]], requires_grad=False).to(device)
            old_uc = torch.tensor(user_count[batch_train[:, 0]], requires_grad=False).to(device)
            old_mc = torch.tensor(movie_count[batch_train[:, 1]], requires_grad=False).to(device)

        user_policy_context =  torch.reshape(userPop, (-1, 1))
        user_adj_prediction = user_policy.predict(user_policy_context.detach().cpu().numpy())
        #print('user_adj_prediction', user_adj_prediction)
        user_adj_prediction = torch.tensor(user_adj_prediction, requires_grad=False).to(device)
        mask = old_uw != len(Emb_Size) - 1
        user_adj = mask * user_adj_prediction
        new_uw = old_uw + user_adj

        if Dy_Emb_Num == 2:
            movie_policy_context = torch.reshape(moviePop, (-1, 1))
            movie_adj_prediction = movie_policy.predict(movie_policy_context.detach().cpu().numpy())
            movie_adj_prediction = torch.tensor(movie_adj_prediction, requires_grad=False).to(device)
            mask = old_mw != len(Emb_Size) - 1
            movie_adj = mask * movie_adj_prediction
            new_mw = old_mw + movie_adj
        else:
            new_mw = 0
        
        with torch.no_grad():
            
            temp_emb_user = model.emb_user

            for i in range(len(Emb_Size) - 1):
                j = i + 1
                part_userId = userId[
                    ((old_uw == i) * (new_uw == j)).nonzero().squeeze(1)]
                if len(part_userId) > 0:
                    model.emb_user[j].weight[part_userId, :] = model.W_user[i](
                        model.emb_user[i].weight[part_userId, :])

            if Dy_Emb_Num == 2:
                temp_emb_movie = model.emb_movie

                for i in range(len(Emb_Size) - 1):
                    j = i + 1
                    part_movieId = movieId[
                        ((old_mw == i) * (new_mw == j)).nonzero().squeeze(1)]
                    if len(part_movieId) > 0:
                        model.emb_movie[j].weight[part_movieId, :] = model.W_movie[i](
                            model.emb_movie[i].weight[part_movieId, :])
            
            rating = model(new_uw, new_mw, userId, movieId, movie_vec)
            #rating = rating.squeeze(1).squeeze(1) if Loss_Type == 'CrossEntropyLoss' else rating.squeeze(1)
            batch_losses = Batch_Losses(Loss_Type, rating, batch_train_target)
            rewards = 1 - batch_losses
        model.emb_user = temp_emb_user


        if Dy_Emb_Num == 1:
            if Reward_Base == 'last_loss':
                baseline = 1 - old_ul[:, 0]
                rewards = rewards - baseline
            elif Reward_Base == 'ave_loss':
                last_num_tensor = torch.Tensor([last_num]).repeat(len(old_uc)).to(device)
                baseline = 1 - torch.sum(old_utl, dim=1) / torch.where(old_uc < last_num, old_uc, last_num_tensor)
            rewards = rewards - baseline
            threshold = torch.tensor([0.0]).to(device)
            rewards = (rewards > threshold).int()
            
            #if index % 100000 == 0:
            #    print("rewards: ", rewards[:50].tolist())   
        
            user_policy_context = user_policy_context.detach().cpu().numpy()
            new_uw = new_uw.detach().cpu().numpy()
            user_adj_prediction = user_adj_prediction.detach().cpu().numpy()
            rewards = rewards.detach().cpu().numpy()
            user_policy.fit(user_policy_context, new_uw, rewards, warm_start=True)


        elif Dy_Emb_Num == 2:
            model.emb_movie = temp_emb_movie
            if Reward_Base == 'last_loss':
                baseline_u = 1 - old_ul[:, 0]
                baseline_m = 1 - old_ml[:, 0]

            elif Reward_Base == 'ave_loss':
                last_num_tensor = torch.Tensor([last_num]).repeat(len(old_uc)).to(device)
                baseline_u = 1 - torch.sum(old_utl, dim=1) / torch.where(old_uc < last_num, old_uc, last_num_tensor)
                baseline_m = 1 - torch.sum(old_mtl, dim=1) / torch.where(old_mc < last_num, old_mc, last_num_tensor)
            
            rewards_u = rewards - baseline_u
            rewards_m = rewards - baseline_m
            threshold = torch.tensor([0.0]).to(device)
            rewards_u = (rewards_u > threshold).int()
            rewards_m = (rewards_m > threshold).int()

            #if index % 100000 == 0:
            #    print("rewards_u: ", rewards_u[:50].tolist())
            #    print("rewards_m: ", rewards_m[:50].tolist())

            user_policy_context = user_policy_context.detach().cpu().numpy()
            new_uw = new_uw.detach().cpu().numpy()
            rewards_u = rewards_u.detach().cpu().numpy()
            movie_policy_context = movie_policy_context.detach().cpu().numpy()
            new_mw = new_mw.detach().cpu().numpy()
            rewards_m = rewards_m.detach().cpu().numpy()
            user_adj_prediction = user_adj_prediction.detach().cpu().numpy()
            movie_adj_prediction = movie_adj_prediction.detach().cpu().numpy()
            user_policy.fit(user_policy_context, new_uw, rewards_u)
            movie_policy.fit(movie_policy_context, new_uw, rewards_m)



def update_RS_Old(index, features, Len_Features, target, mode, Setting_Movie_Popularity, Setting_User_Popularity):
    """ Update RS's embeddings and NN """
    global train_sample_loss, train_sample_accuracy, user_dims_record, movie_dims_record
    index_end = index + Batch_Size
    if index_end >= Len_Features:
        batch_train = features[index:Len_Features]
        batch_train_target = target[index:Len_Features]
    else:
        batch_train = features[index:index_end]
        batch_train_target = target[index:index_end]

    userId = torch.tensor(batch_train[:, 0], requires_grad=False).to(device)
    movieId = torch.tensor(batch_train[:, 1], requires_grad=False).to(device)
    userPop = torch.tensor(batch_train[:, 2], requires_grad=False).to(device)
    moviePop = torch.tensor(batch_train[:, 3], requires_grad=False).to(device)
    old_uw = torch.tensor(user_weights[batch_train[:, 0]], requires_grad=False).to(device)
    old_mw = torch.tensor(movie_weights[batch_train[:, 1]], requires_grad=False).to(device)
    old_ul = torch.tensor(user_losses[batch_train[:, 0], :], requires_grad=False).to(device)
    old_ml = torch.tensor(user_losses[batch_train[:, 1], :], requires_grad=False).to(device)
    movie_vec = torch.tensor([genome_scores_dict[str(batch_train[:, 1][i])] for i in range(len(batch_train[:, 1]))],
                             requires_grad=False).to(device) if Dy_Emb_Num == 1 else None
    if movie_vec is not None:
        movie_vec = movie_vec.to(torch.float32)
    batch_train_target = torch.tensor(batch_train_target,
                                      dtype=torch.int64 if Loss_Type == 'CrossEntropyLoss' else torch.float32,
                                      requires_grad=False).to(device)

    with torch.no_grad():
        #batch_size x Emb_Size
        #user_nw = user_policy(userPop, old_uw, old_ul)
        #user_nw_prob = nn.functional.softmax(user_nw, dim=-1)
        #prevent that new_uw exceeds the limitation of len(Emb_Size)
        #new_uw = torch.argmax(user_nw_prob, dim=1)

        #old_uw_one_hot = nn.functional.one_hot(old_uw, num_classes=len(Emb_Size))
        #user_policy_context = torch.cat((old_uw_one_hot, userPop/max_user_popularity), 1)

        #Linear ladder bandit
        user_policy_context =  torch.reshape(userPop, (-1, 1))
        user_adj_prediction = user_policy.predict(user_policy_context.detach().cpu().numpy())
        user_adj_prediction = torch.tensor(user_adj_prediction, requires_grad=False).to(device)
        mask = old_uw != len(Emb_Size) - 1
        user_adj = mask * user_adj_prediction
        new_uw = old_uw + user_adj
        

        for i in range(len(Emb_Size) - 1):
            j = i + 1
            part_userId = userId[((old_uw == i) * (new_uw == j)).nonzero().squeeze(1)]
            if len(part_userId) > 0:
                model.emb_user[j].weight[part_userId, :] = model.W_user[i](
                    model.emb_user[i].weight[part_userId, :])
        
        if Dy_Emb_Num == 2:
            movie_policy_context = torch.reshape(moviePop, (-1, 1))
            movie_adj_prediction = movie_policy.predict(movie_policy_context.detach().cpu().numpy())
            movie_adj_prediction = torch.tensor(movie_adj_prediction, requires_grad=False).to(device)
            mask = old_mw != len(Emb_Size) - 1
            movie_adj = mask * movie_adj_prediction
            new_mw = old_mw + movie_adj
            
            for i in range(len(Emb_Size) - 1):
                j = i + 1
                part_movieId = movieId[
                    ((old_mw == i) * (new_mw == j)).nonzero().squeeze(1)]
                if len(part_movieId) > 0:
                    model.emb_movie[j].weight[part_movieId, :] = model.W_movie[i](
                        model.emb_movie[i].weight[part_movieId, :])
            
        else:

            new_mw = 0

        
        #if index % 50000 == 0:
        #    print("old_uw", old_uw)
        #    print("new_uw", new_uw)
        

    rating = model(new_uw, new_mw, userId, movieId, movie_vec)
    #rating = rating.squeeze(1).squeeze(1) if Loss_Type == 'CrossEntropyLoss' else rating.squeeze(1)
    

    # batch_losses: (batch_size)
    batch_losses = Batch_Losses(Loss_Type, rating, batch_train_target)
    loss = sum(batch_losses)
    batch_accuracies = Batch_Accuracies(Loss_Type, rating, batch_train_target)

    train_sample_loss += list(batch_losses.detach().cpu().numpy())
    losses[mode].append(loss.detach().cpu().numpy())
    train_sample_accuracy += batch_accuracies
    accuracies[mode].append((sum(batch_accuracies), len(batch_train_target)))

    user_dims_record += [Emb_Size[item] for item in new_uw.detach().cpu()]
    if Dy_Emb_Num == 2:
        movie_dims_record += [Emb_Size[item] for item in new_mw.detach().cpu()]

    if Train_Method == 'AutoML':
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()
    elif Train_Method == 'Supervised':
        optimizer_whole.zero_grad()
        loss.backward()
        optimizer_whole.step()
    else:
        print('No such Train_Method')

    user_weights[batch_train[:, 0]] = new_uw.detach().cpu().numpy()
    #print('user memory example', Emb_Size[user_weights][:100])
    #print('user memory', np.sum(Emb_Size[user_weights]))
    global user_embedding_table, movie_embedding_table
    user_embedding_table = np.append(user_embedding_table, np.sum(Emb_Size[user_weights]))
    movie_weights[batch_train[:, 1]] = new_mw.detach().cpu().numpy() if Dy_Emb_Num == 2 else np.zeros((len(batch_train),))
    #print('movie memory', np.sum(Emb_Size[movie_weights]))
    movie_embedding_table = np.append(movie_embedding_table, np.sum(Emb_Size[movie_weights]))
    user_losses[batch_train[:, 0], :] = np.reshape(batch_losses.detach().cpu().numpy(), (-1, 1))
    movie_losses[batch_train[:, 1], :] = np.reshape(batch_losses.detach().cpu().numpy(), (-1, 1))
    final_user_pop[batch_train[:, 0]] = batch_train[:, 2]
    final_user_pop[batch_train[:, 1]] = batch_train[:, 3]

    if Reward_Base == 'ave_loss':
        user_total_losses[batch_train[:, 0], 1:last_num] = user_total_losses[batch_train[:, 0], 0:last_num-1]
        movie_total_losses[batch_train[:, 1], 1:last_num] = movie_total_losses[batch_train[:, 1], 0:last_num-1]
        user_total_losses[batch_train[:, 0], 0] = user_losses[batch_train[:, 0], 0]
        movie_total_losses[batch_train[:, 1], 0] = movie_losses[batch_train[:, 1], 0]

        user_count[batch_train[:, 0]] += 1
        movie_count[batch_train[:, 1]] += 1

if __name__ == "__main__":
    Output_Dim = 5 if Loss_Type == 'CrossEntropyLoss' else 1
    
    train_features, test_features, train_target, test_target, genome_scores_dict, \
    train_feature_data, test_feature_data, Len_Train_Features, Len_Test_Features, \
    User_Num, Movie_Num, max_user_popularity, max_movie_popularity = load_data()
    train_feature_data, test_feature_data = train_feature_data[:Len_Train_Features], test_feature_data[:Len_Test_Features]
    """
    train_features, test_features, train_target, test_target, \
    train_feature_data, test_feature_data, Len_Train_Features, Len_Test_Features, \
    User_Num, Movie_Num, max_user_popularity, max_movie_popularity = load_data_Amazon()
    train_feature_data, test_feature_data = train_feature_data[:Len_Train_Features], test_feature_data[:Len_Test_Features]
    """
    # hyperparameter preparation for constructing RL-based policy
    Setting_User_Popularity = [max_user_popularity, 32]
    Setting_Movie_Popularity = [max_movie_popularity, 32]
    Setting_User_Weight = [User_Num, len(Emb_Size)]
    Setting_Movie_Weight = [Movie_Num, len(Emb_Size)]
    
    if Train_Method == 'AutoML' and H_alpha > 0:
        Beta_Dis = nn.Embedding(num_embeddings=max(max_user_popularity, max_movie_popularity), embedding_dim=len(Emb_Size)).to(dtype=torch.float32)
        Beta_Dis.weight.data = torch.tensor(np.array([Beta(len(Emb_Size), popularity, Beta_Beta) for popularity in range(1, max(max_user_popularity, max_movie_popularity) + 1)]), dtype=torch.float32, requires_grad=False)
        Beta_Dis.weight.requires_grad = False
        Beta_Dis.to(device)
        criterion = nn.KLDivLoss(reduction='sum')

    #TODO: construct context bandit based policy
    base_algorithm = LogisticRegression(solver='lbfgs', warm_start=True)
    beta_prior = ((3./len(Emb_Size), 4), 2) # until there are at least 2 observations of each class, will use this prior
    beta_prior_ucb = ((5./len(Emb_Size), 4), 2) # UCB gives higher numbers, thus the higher positive prior
    beta_prior_ts = ((2./np.log2(Emb_Size), 4), 2)
    user_policy = LinUCB(nchoices=2, beta_prior=None, alpha=args.alpha, ucb_from_empty=False, random_state=1111)
    movie_policy = LinUCB(nchoices=2, beta_prior=None, alpha=args.alpha, ucb_from_empty=False, random_state=1111)
    #user_policy = neural_bandit(setting_popularity=Setting_User_Popularity, lr=0.0001, nu=0.01)
    #movie_policy = neural_bandit(setting_popularity=Setting_Movie_Popularity, lr=0.0001, nu=0.01)
    #Construct RS MLP
    #model = RS_MLP_New(Output_Dim, Dy_Emb_Num)
    model = RS_MLP_Old(Output_Dim, Dy_Emb_Num)
    model.to(device)

    if Model_Gpu:
        print('\n========================================================================================\n')
        print('Model_Gpu?:', next(model.parameters()).is_cuda)
        print('Memory:    ', torch.cuda.memory_allocated(0) / 1024 ** 3, 'GB', torch.cuda.memory_cached(0) / 1024 ** 3, 'GB')
        print('\n========================================================================================\n')
    
    print('User_Num', User_Num)
    print('Movie_Num', Movie_Num)
    user_weights = np.zeros((User_Num,), dtype=np.int64)
    movie_weights = np.zeros((Movie_Num,), dtype=np.int64)
    user_embedding_table = np.array([])
    movie_embedding_table = np.array([])
    final_user_pop = np.zeros((User_Num,), dtype=np.int64)
    final_movie_pop = np.zeros((Movie_Num,), dtype=np.int64)
    user_losses = np.ones((User_Num, 1), dtype=np.float32)
    movie_losses = np.ones((Movie_Num, 1), dtype=np.float32)
    
    if Reward_Base == 'ave_loss':
        user_total_losses = np.zeros((User_Num, last_num), dtype=np.float32)
        movie_total_losses = np.zeros((Movie_Num, last_num), dtype=np.float32)
        user_count = np.zeros((User_Num,), dtype=np.float32)
        movie_count = np.zeros((Movie_Num,), dtype=np.float32)

    t0 = time.time()
    # Optimizers
    optimizer_model = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_model, weight_decay=0)

    losses = {'train':[], 'test':[]}
    accuracies = {'train':[], 'test':[]}
    train_sample_loss = list()
    train_sample_accuracy = list()
    user_dims_record = list()
    movie_dims_record = list()

    print('\n******************************************Train******************************************\n')

    for epoch_i in range(Epoch):
        index = 0
        while index < Len_Train_Features:
            update_ladder_bandit_controller(index, train_features, train_target, Setting_Movie_Popularity=Setting_Movie_Popularity, Setting_User_Popularity=Setting_User_Popularity)
            update_RS_Old(index, train_features, Len_Train_Features, train_target, mode='train', Setting_Movie_Popularity=Setting_Movie_Popularity, Setting_User_Popularity=Setting_User_Popularity)

            if len(losses['train']) % 10 == 0:
                print('Epoch = {:>3}  Batch = {:>4}/{:>4} ({:.3f}%)    train_loss = {:.3f}     train_accuracy = {:.3f}     total_time = {:.3f} min'.format(
                    epoch_i, index + Batch_Size, Len_Train_Features, 100 * (index + Batch_Size) / Len_Train_Features, sum(losses['train'][-10:]) / 10,
                    sum([item[0] / item[1] for item in accuracies['train'][-10:]]) / 10,
                    (time.time() - t0) / 60))
            index += Batch_Size

    #############################test#############################
    print('\n******************************************Test******************************************\n')
    t0 = time.time()
    index = 0
    # Len_Test_Features = 20000
    while index < Len_Test_Features:
        update_ladder_bandit_controller(index, test_features, test_target, Setting_Movie_Popularity=Setting_Movie_Popularity, Setting_User_Popularity=Setting_User_Popularity)
        update_RS_Old(index, test_features, Len_Test_Features, test_target, mode='test', Setting_Movie_Popularity=Setting_Movie_Popularity, Setting_User_Popularity=Setting_User_Popularity)

        if len(losses['test']) % 10 == 0:
            print(
                'Test   Batch = {:>4}/{:>4} ({:.3f}%)     test_loss = {:.3f}     test_accuracy = {:.3f}     whole_time = {:.3f} min'.format(
                    index + Batch_Size, Len_Test_Features, 100 * (index + Batch_Size) / Len_Test_Features,
                    sum(losses['test'][-10:]) / 10,
                    sum([item[0] / item[1] for item in accuracies['test'][-10:]]) / 10, (time.time() - t0) / 60))

        index += Batch_Size

    correct_num = sum([item[0] for item in accuracies['train']])
    train_num = sum([item[1] for item in accuracies['train']])

    print('Train Loss: {:.4f}'.format(sum(losses['train']) / train_num))

    print('Train Correct Num: {}'.format(correct_num))
    print('Train Num: {}'.format(train_num))

    print('Train Accuracy: {:.4f}'.format(correct_num / train_num))

    correct_num = sum([item[0] for item in accuracies['test']])
    test_num = sum([item[1] for item in accuracies['test']])

    print('Test Loss: {:.4f}'.format(sum(losses['test']) / test_num))

    print('Test Correct Num: {}'.format(correct_num))
    print('Test Num: {}'.format(test_num))

    print('Test Accuracy: {:.4f}'.format(correct_num / test_num))

    all_correct_num = sum([item[0] for item in accuracies['train']]) + sum([item[0] for item in accuracies['test']])
    all_num = sum([item[1] for item in accuracies['train']]) + sum([item[1] for item in accuracies['test']])

    print('all Loss: {:.4f}'.format((sum(losses['train']) + sum(losses['test']))/ all_num))

    print('all Correct Num: {}'.format(all_correct_num))
    print('all Num: {}'.format(all_num))

    print('all Accuracy: {:.4f}'.format(all_correct_num / all_num))
   

    #save model
    save_model_name = './save_model/MEDIA_DyEmbNum{}_LossType{}_Reward_Base{}_last{}_TestAcc{:.4f}'.format(
            Dy_Emb_Num, Loss_Type, Reward_Base, last_num,
            correct_num / test_num)    
    torch.save(model.state_dict(), save_model_name + '.pt')
    with open(save_model_name + '_weights.pkl', 'wb') as f:
        if Dy_Emb_Num == 1:
            pk.dump((final_user_pop, user_weights), f)
        elif Dy_Emb_Num == 2:
            pk.dump(((final_user_pop, user_weights), (final_movie_pop, movie_weights)), f)   

    print('user_embedding_table[0]', user_embedding_table[0])
    print('user_embedding_table[-1]', user_embedding_table[-1])
    print('user_embedding_table mean', np.mean(user_embedding_table))
    print('movie_embedding_table[0]', movie_embedding_table[0])
    print('movie_embedding_table[-1]', movie_embedding_table[-1])
    print('movie_embedding_table mean', np.mean(movie_embedding_table))

    with open(save_model_name + '_embedding_table.pkl', 'wb') as f:
        if Dy_Emb_Num == 1:
            pk.dump(user_embedding_table, f)
        elif Dy_Emb_Num == 2:
            pk.dump((user_embedding_table, movie_embedding_table), f)   

    print('Model saved to ' + save_model_name + '.pt')
    print('Weights saved to ' + save_model_name + '_weights.pkl')
    print('Embedding Saved to' + save_model_name + '_embedding_table.pkl')

    feature_data = pd.concat([train_feature_data, test_feature_data])
    print("feature_data: ", feature_data.shape[0], feature_data.shape[1])

    feature_data['user_dims'] = pd.DataFrame(
        [[i] for i in user_dims_record])
    if Dy_Emb_Num == 2:
        feature_data['movie_dims'] = pd.DataFrame([[i] for i in movie_dims_record])
    feature_data['{}_loss_{}'.format(Train_Method[0], Emb_Size)] = pd.DataFrame(
        [[i] for i in train_sample_loss])
    feature_data['{}_acc_{}'.format(Train_Method[0], Emb_Size)] = pd.DataFrame(
        [[i] for i in train_sample_accuracy])

    print('\n****************************************************************************************\n')

    if Model_Gpu:
        print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
        print('Memory:    ', torch.cuda.memory_allocated(0) / 1024 ** 3, 'GB', torch.cuda.memory_cached(0) / 1024 ** 3, 'GB')
        # torch.cuda.empty_cache()
        print('Memory:    ', torch.cuda.memory_allocated(0) / 1024 ** 3, 'GB', torch.cuda.memory_cached(0) / 1024 ** 3, 'GB')
        print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

    Parameter_Name = 'DataSet{}_ValType{}_DyEmbNum{}_LossType{}_RewardBase{}'.format(
        DATA_SET,
        Val_Type if Train_Method == 'AutoML' else 'None',
        Dy_Emb_Num,
        Loss_Type,
        Reward_Base)

    feature_data.to_csv('./results/MEDIA_feature_data_with_loss_{}.csv'.format(Parameter_Name), index=None)

    if Dy_Emb_Num == 1:
        result_user = []
        for i in range(1, 100):
            feature_data1 = feature_data[feature_data['user_frequency'] == i]
            result_user.append(list(feature_data1.mean(axis=0)) + [len(feature_data1)])
        Head = list(feature_data.columns) + ['count']
        pd.DataFrame(result_user).to_csv('./results/MEDIA_result_{}_user.csv'.format(Parameter_Name), index=None,
                                          header=Head)

    elif Dy_Emb_Num == 2:
        result_user, result_movie = [], []
        for i in range(1, 100):
            feature_data1 = feature_data[feature_data['user_frequency'] == i]
            result_user.append(list(feature_data1.mean(axis=0)) + [len(feature_data1)])
        Head = list(feature_data.columns) + ['count']
        pd.DataFrame(result_user).to_csv('./results/MEDIA_result_{}_user.csv'.format(Parameter_Name), index=None,
                                          header=Head)

        for i in range(1, 100):
            feature_data1 = feature_data[feature_data['movie_frequency'] == i]
            result_movie.append(list(feature_data1.mean(axis=0)) + [len(feature_data1)])
        Head = list(feature_data.columns) + ['count']
        pd.DataFrame(result_movie).to_csv('./results/MEDIA_result_{}_movie.csv'.format(Parameter_Name), index=None,
                                          header=Head)


    result = []
    for i in range(int(Train_Size/1000000)):
        feature_data1 = feature_data[i*1000000:(i+1)*1000000]
        result.append(list(feature_data1.mean(axis=0)) + [len(feature_data1)])

    Head = list(feature_data.columns) + ['count']
    pd.DataFrame(result).to_csv('./results/MEDIA_result_{}_trendency.csv'.format(Parameter_Name), index=None, header=Head)

    print('\n****************************************************************************************\n')
    print('os.getpid():   ', os.getpid())
    if torch.cuda.is_available():
        print('torch.cuda:    ', torch.cuda.is_available(), torch.cuda.current_device(), torch.cuda.device_count(), torch.cuda.get_device_name(0), torch.cuda.device(torch.cuda.current_device()))
    else:
        print('GPU is not available!!!')
    print('Train_Size:    ', Train_Size)
    print('Test_Size:     ', Test_Size)
    print('Emb_Size:      ', Emb_Size)
    print('Dy_Emb_Num:    ', Dy_Emb_Num)
    print('Loss_Type:     ', Loss_Type)
    print('Train_Method:  ', Train_Method)
    print('Val_Type:      ', Val_Type)
    print('Beta_Beta:     ', Beta_Beta)
    print('H_alpha:       ', H_alpha)
    print('LR_model:      ', LR_model)
    print('LR_darts:      ', LR_darts)
    print('\n****************************************************************************************\n')
    print('{} done'.format(Train_Method))


