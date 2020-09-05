import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None

    def fit(self, X):
        # 随机初始化聚类中心
        n_samples, n_features = X.shape
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx]

        for _ in range(self.max_iters):
            # 分配样本到最近的聚类中心
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)

            # 更新聚类中心
            new_centroids = np.array([X[self.labels == k].mean(axis=0)
                                    for k in range(self.n_clusters)])

            # 检查收敛
            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

        return self

class GaussianMixture:
    def __init__(self, n_components=3, max_iters=100):
        self.n_components = n_components
        self.max_iters = max_iters

    def fit(self, X):
        n_samples, n_features = X.shape

        # 初始化参数
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covs = np.array([np.eye(n_features) for _ in range(self.n_components)])

        for _ in range(self.max_iters):
            # E步：计算后验概率
            resp = self._e_step(X)

            # M步：更新参数
            self._m_step(X, resp)

        return self

    def _e_step(self, X):
        """计算每个样本属于每个组件的后验概率"""
        resp = np.zeros((X.shape[0], self.n_components))

        for k in range(self.n_components):
            resp[:, k] = self.weights[k] * self._gaussian_pdf(X, self.means[k], self.covs[k])

        resp /= resp.sum(axis=1, keepdims=True)
        return resp

    def _m_step(self, X, resp):
        """更新模型参数"""
        resp_sum = resp.sum(axis=0)

        # 更新权重
        self.weights = resp_sum / X.shape[0]

        # 更新均值
        self.means = np.dot(resp.T, X) / resp_sum[:, np.newaxis]

        # 更新协方差
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covs[k] = np.dot(resp[:, k] * diff.T, diff) / resp_sum[k]

    def _gaussian_pdf(self, X, mean, cov):
        """计算多维高斯分布的概率密度"""
        n = X.shape[1]
        diff = X - mean
        return np.exp(-0.5 * np.sum(np.dot(diff, np.linalg.inv(cov)) * diff, axis=1)) / \
               np.sqrt((2 * np.pi) ** n * np.linalg.det(cov))

class HierarchicalClustering:
    def __init__(self, n_clusters=3, linkage_method='ward'):
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method

    def fit_predict(self, X):
        # 计算距离矩阵
        distances = pdist(X)

        # 构建层次结构
        Z = linkage(distances, method=self.linkage_method)

        # 获取聚类标签
        labels = fcluster(Z, self.n_clusters, criterion='maxclust')
        return labels - 1  # 使标签从0开始

class AnomalyDetector:
    def __init__(self, contamination=0.1):
        self.contamination = contamination

    def fit_predict(self, X):
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 计算马氏距离
        mean = np.mean(X_scaled, axis=0)
        cov = np.cov(X_scaled.T)
        inv_covmat = np.linalg.inv(cov)

        mahalanobis_dist = []
        for x in X_scaled:
            diff = x - mean
            dist = np.sqrt(diff.dot(inv_covmat).dot(diff))
            mahalanobis_dist.append(dist)

        # 根据距离确定阈值
        threshold = np.percentile(mahalanobis_dist,
                                (1 - self.contamination) * 100)

        return np.array(mahalanobis_dist) > threshold

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def fit(self, X, epochs=100, batch_size=32):
        optimizer = optim.Adam(self.parameters())
        criterion = nn.MSELoss()

        X_tensor = torch.FloatTensor(X)
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                batch = X_tensor[i:i+batch_size]

                optimizer.zero_grad()
                output = self(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()

class DeepBeliefNetwork(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.rbm_layers = nn.ModuleList([
            RestrictedBoltzmannMachine(layer_sizes[i], layer_sizes[i+1])
            for i in range(len(layer_sizes)-1)
        ])

    def forward(self, x):
        for rbm in self.rbm_layers:
            x = rbm(x)
        return x

    def pretrain(self, X, epochs=10):
        X_tensor = torch.FloatTensor(X)
        for rbm in self.rbm_layers:
            rbm.train(X_tensor, epochs)
            X_tensor = rbm.sample_hidden(X_tensor)

class RestrictedBoltzmannMachine(nn.Module):
    def __init__(self, visible_dim, hidden_dim):
        super().__init__()
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.weights = nn.Parameter(torch.randn(visible_dim, hidden_dim) * 0.1)
        self.visible_bias = nn.Parameter(torch.zeros(visible_dim))
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_dim))

    def sample_hidden(self, visible):
        hidden_activations = torch.matmul(visible, self.weights) + self.hidden_bias
        hidden_probs = torch.sigmoid(hidden_activations)
        return torch.bernoulli(hidden_probs)

    def sample_visible(self, hidden):
        visible_activations = torch.matmul(hidden, self.weights.t()) + self.visible_bias
        visible_probs = torch.sigmoid(visible_activations)
        return torch.bernoulli(visible_probs)

    def train(self, X, epochs, learning_rate=0.01):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for _ in range(epochs):
            # Positive phase
            pos_hidden = self.sample_hidden(X)
            pos_associations = torch.matmul(X.t(), pos_hidden)

            # Negative phase
            neg_visible = self.sample_visible(pos_hidden)
            neg_hidden = self.sample_hidden(neg_visible)
            neg_associations = torch.matmul(neg_visible.t(), neg_hidden)

            # Update parameters
            update = pos_associations - neg_associations
            optimizer.zero_grad()
            self.weights.data += learning_rate * update

class HebbianLearning:
    def __init__(self, input_dim, output_dim, learning_rate=0.01):
        self.weights = np.random.randn(input_dim, output_dim) * 0.1
        self.learning_rate = learning_rate

    def fit(self, X, epochs=100):
        for _ in range(epochs):
            for x in X:
                x = x.reshape(-1, 1)
                # 计算输出
                y = np.dot(x.T, self.weights)
                # Hebbian更新规则
                self.weights += self.learning_rate * np.dot(x, y)

    def transform(self, X):
        return np.dot(X, self.weights)

class GAN:
    def __init__(self, latent_dim, data_dim):
        self.latent_dim = latent_dim
        self.data_dim = data_dim

        # 定义生成器
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, data_dim),
            nn.Tanh()
        )

        # 定义判别器
        self.discriminator = nn.Sequential(
            nn.Linear(data_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.g_optimizer = optim.Adam(self.generator.parameters())
        self.d_optimizer = optim.Adam(self.discriminator.parameters())
        self.criterion = nn.BCELoss()

    def train(self, X, epochs=100, batch_size=32):
        X_tensor = torch.FloatTensor(X)

        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                batch = X_tensor[i:i+batch_size]

                # 训练判别器
                self.d_optimizer.zero_grad()

                # 真实数据的损失
                real_labels = torch.ones(batch.size(0), 1)
                real_outputs = self.discriminator(batch)
                d_loss_real = self.criterion(real_outputs, real_labels)

                # 生成数据的损失
                z = torch.randn(batch.size(0), self.latent_dim)
                fake_data = self.generator(z)
                fake_labels = torch.zeros(batch.size(0), 1)
                fake_outputs = self.discriminator(fake_data.detach())
                d_loss_fake = self.criterion(fake_outputs, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()

                # 训练生成器
                self.g_optimizer.zero_grad()
                fake_outputs = self.discriminator(fake_data)
                g_loss = self.criterion(fake_outputs, real_labels)
                g_loss.backward()
                self.g_optimizer.step()

class SOM:
    def __init__(self, map_size=(10,10), input_dim=2, learning_rate=0.1):
        self.map_size = map_size
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.weights = np.random.randn(map_size[0], map_size[1], input_dim)

    def find_bmu(self, x):
        """找到最佳匹配单元(Best Matching Unit)"""
        distances = np.sum((self.weights - x) ** 2, axis=2)
        return np.unravel_index(np.argmin(distances), self.map_size)

    def neighborhood_function(self, bmu, sigma=1.0):
        """计算邻域函数"""
        y, x = np.ogrid[0:self.map_size[0], 0:self.map_size[1]]
        distances = ((x - bmu[1]) ** 2 + (y - bmu[0]) ** 2) / (2 * sigma ** 2)
        return np.exp(-distances)

    def fit(self, X, epochs=100, sigma=1.0):
        for epoch in range(epochs):
            for x in X:
                # 找到BMU
                bmu = self.find_bmu(x)

                # 计算邻域
                neighborhood = self.neighborhood_function(bmu, sigma)

                # 更新权重
                learning = self.learning_rate * neighborhood[:, :, np.newaxis]
                self.weights += learning * (x - self.weights)

            # 降低学习率和sigma
            self.learning_rate *= 0.95
            sigma *= 0.95